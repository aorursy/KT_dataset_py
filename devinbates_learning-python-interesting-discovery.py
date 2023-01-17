from time import time, sleep

def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1 = time()
    fn(arg)
    t2 = time()
    return t2 - t1
def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1 = time()
    fn(arg)
    t2 = time()
    return t2 - t1

timeforfunction = time_call(sleep, 5)

t1 = time()
sleep(5)
t2 = time()
timetaken = t2 - t1

print("there is a loss of" , timeforfunction - timetaken, "seconds")