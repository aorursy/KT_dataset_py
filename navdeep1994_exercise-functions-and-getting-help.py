# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

     return round(num,2)



q1.check()
# Uncomment the following for a hint

q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
def round_to_two_places(num):

     return round(num,2)



q1.check()
#q2.solution()
def to_smash(total_candies,total_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % total_friends



q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = abs(-10)

y = 5

# # Which of the two variables above has the smallest absolute value?

smallest_abs = min(x, y)
def f(x):

    y = abs(x)

    return y



print(f(5))