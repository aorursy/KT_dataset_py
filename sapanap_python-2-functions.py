### Example 

def least_difference(a, b, c):

    """Return the smallest difference between any two numbers

    among a, b and c.

    

    >>> least_difference(1, 5, -5)

    4

    """

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)
help(least_difference)
least_difference(4,8,10)
### Example 

mystery = print()

print(mystery)
# Example of Our own functions

def mult_by_five(x):

    return 5 * x



def call(fn, arg):

    """Call fn on arg"""

    return fn(arg)



def squared_call(fn, arg):

    """Call fn on the result of calling fn on arg"""

    return fn(fn(arg))



print(

    call(mult_by_five, 2),

    squared_call(mult_by_five, 3)

)
# Example of python max() function



def mod_5(x):

    """Return the remainder of x after dividing by 5"""

    return x % 5



print(

    'Which number is biggest?',

    max(100, 51, 14),

    'Which number is the biggest modulo 5?',

    max(100, 51, 14, key=mod_5),

    sep='\n',

)
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    return round(num, 2)



print(round_to_two_places(12.34534534))
def to_smash(total_candies, number_of_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % number_of_friends



print(to_smash(433))

print(to_smash(433,5))
from time import time

from time import sleep

t = time()

print(t, "seconds since the Epoch")



duration = 5

print("Getting sleepy. See you in", duration, "seconds")

sleep(duration)

print("I'm back. What did I miss?")
from time import time

from time import sleep

def time_call(fn, arg=3):

    """Return the amount of time the given function takes (in seconds) when called with the given argument.

    """

    t0 = time()

    fn(arg)

    t1 = time()

    elapsed = t1 - t0

    return elapsed



time_gone = time_call(sleep,3)

print("Time lapsed in seconds = ",time_gone)

print("Time lapsed in seconds = ", time_call(sleep,3))
def slowest_call(fn, arg1, arg2, arg3):

    """Return the amount of time taken by the slowest of the following function

    calls: fn(arg1), fn(arg2), fn(arg3)

    """

    return min(time_call(fn,arg1),time_call(fn,arg2),time_call(fn,arg3))

print(slowest_call(sleep,9, 6, 10))