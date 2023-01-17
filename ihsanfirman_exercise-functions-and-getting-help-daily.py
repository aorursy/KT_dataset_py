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

    pass

    return round(num,2)

q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

round(31234, ndigits=-2)
q2.solution()
def to_smash(total_candies, total_friends = 3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % total_friends



q3.check()
#q3.hint()
#q3.solution()
round(9.9999, ndigits = 2)
x = -10

y = 5

 # Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x), abs(y))

smallest_abs
def f(x):

    y = abs(x)

    return y





print(f(5))