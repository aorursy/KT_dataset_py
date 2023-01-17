# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    return round(num,2)

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass

round_to_two_places(3.14159)

# Check your answer

q1.check()
# Put your test code here
# Check your answer (Run this code cell to receive credit!)

q2.solution()
def to_smash(total_candies, n_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> 

    1

    """

    return total_candies % n_friends

to_smash(91)

# Check your answer

q3.check()
#q3.hint()
round_to_two_places(9.9999)
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

smallest_abs = abs(min(x, y))
def f(x):    

    y = abs(x)

    return y



print(f(5))