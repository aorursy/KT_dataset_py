# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):



    return (round(num, 2))



    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

q2.solution()
def to_smash(total_candies, nb_friends = 3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % nb_friends

    

to_smash(100)

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = abs(-10)

y = abs(5)

# # Which of the two variables above has the smallest absolute value?



def smallest_abs(x,y):

    i1 = abs(x)

    i2 = abs(y)

    return min(i1,i2)



smallest_abs(x, y)
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

    tDiff = t1 - t0

    return tDiff

    pass



time_call(sleep, 3)

#q5.hint()

#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):

    """Return the amount of time taken by the slowest of the following function

    calls: fn(arg1), fn(arg2), fn(arg3)

    """

    

    Tmax = max(arg1, arg2, arg3)

    t0 = time()

    fn(Tmax)

    t1 = time()

    tDiff = t1 - t0

    return (tDiff)

    pass



slowest_call(sleep, 1, 5, 3)



#q6.hint()
#q6.solution()