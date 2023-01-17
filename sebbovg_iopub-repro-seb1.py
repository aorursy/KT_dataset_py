import sys

import time
start = time.time()

max_dur = 2*60

chars = 0



def write(s):

    global chars

    s %= chars  # Show previous count.

    chars += len(s)

    sys.stdout.write(s)



write("start [chars: %d]\n")

    

while True:

    now = time.time()

    elapsed = now - start

    remaining = max_dur - elapsed

    if remaining < 0:

        write("\ndone: [chars: %d]\n")

        break



    time.sleep(0.01)



    write("Elapsed: %ds, remaining: %ds  [chars: %%d]        \r" % (elapsed, remaining))

    sys.stdout.flush()
print("chars: %s" % chars)