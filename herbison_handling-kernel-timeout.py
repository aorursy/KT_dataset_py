import signal



def signal_handler(signum, frame):

    raise Exception("Timed out!")



signal.signal(signal.SIGALRM, signal_handler)

signal.alarm(5)   # 5 seconds



print("Five seconds to work:")
import time

print("fast work")

time.sleep(1)




try:

    while True:

        print("real work...")

        time.sleep(1);

except Exception:

    print("failed")