# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    x = round(num,2)

    return x



print(round_to_two_places(1.123456))



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
def round_as_negative(num):

    x = round(num, -1)

    return x



print(round_as_negative(6.4321))
#q2.solution()
def to_smash(total_candies, friends = 3):

    return total_candies % friends



    """Returns the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between any number of friends, with the default argument set as 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % 3



print(to_smash(91))



q3.check()
#q3.hint()
#q3.solution()
# ruound_to_two_places(9.9999) -- not defined; spelled incorrectly



print(round_to_two_places(9.999))
x = -10

y = 5

# # Which of the two variables above has the smallest absolute value?

# smallest_abs = min(abs(x, y)) -- variable, not a function



def smallest_abs(x,y):

    z = min(x,y)

    return z



print(smallest_abs(x,y))
def f(x):

    y = abs(x)

# return y -- outer level indentation

    return y



print(f(5))