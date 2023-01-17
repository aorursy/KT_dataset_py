# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 

# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    return round(num,2)



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

q1.solution()
# Put your test code here

print(round(338424,-3))
q2.solution()
def to_smash(total_candies,num_of_frnds = 3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between n friends. default value of n = 3.

    

    >>> to_smash(91)

    1

    >>> to_smash(91,6)

    1

    """

    return total_candies % num_of_frnds



q3.check()
#q3.hint()
q3.solution()
round_to_two_places(9.9999)
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x),abs(y))

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(5))
# Importing the function 'time' from the module of the same name. 

# (We'll discuss imports in more depth later)

from time import time

t = time()

print(t, "seconds since the Epoch")
from time import sleep

duration = 5

print("Getting sleepy. See you in", duration, "seconds")

sleep(duration)

print("I'm back. What did I miss?")
def time_call(fn, arg):

    """Return the amount of time the given function takes (in seconds) when called with the given argument.

    """

    t0 = time()

    fn(arg)

    t1 = time()

    elapsed = t1 - t0

    return elapsed

    pass



time_call(sleep, 2)
#q5.hint()

q5.solution()
def slowest_call(fn, arg1, arg2, arg3):

    """Return the amount of time taken by the slowest of the following function

    calls: fn(arg1), fn(arg2), fn(arg3)

    """

    t_taken=[]

    t_start=time()

    fn(arg1)

    t_end=time()

    elapsed_1 = t_end - t_start

    t_start=time()

    fn(arg2)

    t_end=time()

    elapsed_2 = t_end - t_start

    t_start=time()

    fn(arg3)

    t_end=time()

    elapsed_3 = t_end - t_start

    t_taken = [elapsed_1,elapsed_2,elapsed_3]

    return min(t_taken)

    pass
q6.hint()
q6.solution()