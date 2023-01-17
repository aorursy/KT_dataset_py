# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    return round(num, 2)



# Check your answer

q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

num = 3141.59

round(num, -3)
# Check your answer (Run this code cell to receive credit!)

q2.solution()
def to_smash(total_candies, num_of_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between num_of_friends friends. num_of_friends default is 3.

    

    >>> to_smash(91)

    1

    """

    return total_candies % num_of_friends



# Check your answer

q3.check()
#q3.hint()
#q3.solution()
def ruound_to_two_places(num):

    return round(num, 2)



ruound_to_two_places(9.9999)
x = -10

y = 5

# # Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x), abs(y))

smallest_abs
def f(x):

    y = abs(x)

    return y



print(f(5))