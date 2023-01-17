# SETUP. You don't need to worry for now about what this code does or how it works.

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

    return round(num, 2)



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

q1.solution()
round(54321.54321, -2)
q2.solution()
def to_smash(total_candies,num_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.Check the number of friends and 

    split the candies by that number

    

    >>> to_smash(91)

    1

    """

    return total_candies % num_friends



q3.check()
#q3.hint()
q3.solution()
round_to_two_places(round(9.9999,2))
x = -10

y = 5

# # Which of the two variables above has the smallest absolute value?

# smallest_abs = min(abs(x, y))

min(abs(x),abs(y))
def f(x):

    y = abs(x)

    print(f"The Absolute value is {y}.")

    return y
# Importing the function 'time' from the module of the same name. 

# (We'll discuss imports in more depth later)

from time import time

t = time()

print(t, "seconds since the Epoch")
from time import sleep

duration =2

print("Getting sleepy. See you in", duration, "seconds")

sleep(duration)

print("I'm back. What did I miss?")
from time import sleep, time

def time_call(fn, arg):

    """Return the amount of time the given function takes (in seconds) when called with the given argument.

    """

    time_start = time()

    #print(f"Start time is {time_start}")

    fn(arg)

    time_end = time()

    total_time = time_end - time_start

    print(f"Total time is {total_time} and the input argument was {arg}.")

    return total_time

time_call(f,5) # Use the working function, f, from above to verify
q5.solution()
from time import sleep, time

def time_call(fn, arg):

    """Return the amount of time the given function takes (in seconds) when called with the given argument.

    """

    time_start = time()

    #print(f"Start time is {time_start}")

    fn(arg)

    time_end = time()

    total_time = time_end - time_start

    print(f"Total time is {total_time} and the input argument was {arg}.")

    return total_time



def my_checker(x):

    sleep(x)

    #print(f"I am now awake inside my_checker where the input was {x} and the output will be {x*2}.")

    return x*2



def slowest_call(fn, arg1, arg2, arg3):

    """Return the amount of time taken by the slowest of the following function

    calls: fn(arg1), fn(arg2), fn(arg3)

    """



    return max(time_call(fn, arg1), time_call(fn, arg2), time_call(fn, arg3))
#I have no idea why this appears to be calling my_checker six times instead of just three times - HELP!

#time_call is being executed three times with the appropriate arguments

#The second time through, the time is being used as the argument (not correct)

#The good news is  that the correct answer is found at the end of the day at the expense of execution time.



slowest_call(my_checker, time_call(my_checker,6),time_call(my_checker,2),time_call(my_checker,4))
#q6.hint()
q6.solution()