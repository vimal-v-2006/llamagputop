#!/usr/bin/env python3
"""
llamagputop web dashboard — the same data layer as the TUI, served as a
browser UI. Stdlib only, like the TUI itself.

Usage:
    python3 webapp.py [PORT]

PORT defaults to 8371. Open http://localhost:8371 — the dashboard polls
/api/state on the same cadence the TUI offers (0.5 / 1 / 2 / 5 s, switchable
in the header). No build step: web/index.html is one self-contained file.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import llamagputop as lp

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# ------------------------------------------------------------------ collector
# One snapshot is taken every tick, exactly the way the TUI's draw loop does it
# (feed thread hands in whatever it has; nothing blocks the HTTP handlers).
LATEST = {
    "at": None,        # monotonic time of the last collect
    "gpus": [], "cpu": None, "mem": None,
    "llamas": [], "cfgs": {}, "procs": [],
}
LOCK = threading.Lock()
START = time.time()
_stop = threading.Event()


def _collector(gpus, feeds):
    lfeed = lp._LlamaFeed(None)
    lfeed.start()
    tick = 0.0
    while not _stop.is_set():
        t0 = time.monotonic()
        try:
            g, c, m, ll, cfgs, procs = lp._collect(gpus, None, lfeed)
            with LOCK:
                LATEST.update(gpus=g, cpu=c, mem=m,
                              llamas=ll, cfgs=cfgs, procs=procs, at=t0)
        except Exception:
            pass
        # sleep out the rest of the tick so collects stay ~1 s apart
        _stop.wait(max(0.1, 1.0 - (time.monotonic() - t0)))
    lfeed.stop = True
    for f in feeds.values():
        f.stop = True


# ------------------------------------------------------------------ endpoints
def _series_json(limit=240):
    """The rolling histories the TUI charts, thinned to `limit` points."""
    out = {}
    for key, s in list(lp.HIST.items()):
        s = list(s)
        if not s:
            continue
        if len(s) > limit:
            step = max(1, len(s) // limit)
            s = s[::step]
        out[key] = s
    return out


def _state_json():
    with LOCK:
        g = [dict(s) for s in LATEST["gpus"]]
        cpu, mem = LATEST["cpu"], LATEST["mem"]
        ll = [dict(d) for d in LATEST["llamas"]]
        cfgs, procs = LATEST["cfgs"], list(LATEST["procs"])
        at = LATEST["at"]
    total_w, _missing = lp.system_power(g, cpu) if g else (None, None)
    pmin, pavg, _pmax = lp._extremes("power_series")
    e_j = lp._session.get("energy_j", 0.0)
    # fleet tokens/s per kW while anyone is generating (the TUI's efficiency cell)
    gen = [d.get("tg") or 0 for d in ll if d.get("phase") == "generating"]
    eff = (sum(gen) / (total_w / 1000.0)) if (gen and total_w) else None
    return {
        "version": "webapp-1",
        "server_started": START,
        "tick_at": at,
        "total_power_w": total_w,
        "power_avg_w": pavg,
        "energy_wh": e_j / 3.6e6,
        "eff_t_per_kw": eff,
        "gpus": g,
        "cpu": cpu,
        "mem": mem,
        "llamas": ll,
        "cfgs": cfgs,
        "procs": procs,
        "hist": _series_json(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "llamagputop-web/1.0"

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        data = body if isinstance(body, bytes) else body.encode()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path == "/":
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, json.dumps(_state_json()), "application/json")
        elif path == "/api/series":
            n = int(parse_qs(u.query).get("limit", ["240"])[0])
            self._send(200, json.dumps(_series_json(n)), "application/json")
        elif path == "/healthz":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, format, *args):
        pass  # keep the console quiet, like the TUI does


def main():
    argv = sys.argv[1:]
    port = int(next((a for a in argv if a.isdigit()), 8371))
    gpus, feeds = lp.discover_gpus()
    for f in feeds.values():
        f.start()
    threading.Thread(target=_collector, args=(gpus, feeds), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(f"llamagputop web dashboard — http://127.0.0.1:{port}")
    print("  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
