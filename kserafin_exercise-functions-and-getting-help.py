# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    return round (num, 2)

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
round(338424, -5)
q2.solution()
def to_smash(total_candies, number_of_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between given number of friends(if number of friends is not provided, it assume 3 friends).

    

    >>> to_smash(91, 4)

    3

    """

    return total_candies % number_of_friends



q3.check()
#q3.hint()
#q3.solution()
def ruound_to_two_places(i):

    """Returns given number rounded to two decimal places after comma"""

    return round(i, 2)

ruound_to_two_places(9.1234)

    
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

def smallest_abs(x, y): 

    return min(abs(x), abs(y))



smallest_abs(x, y)

def f(x):

    """Function returns absolute value of the given number"""

    y = abs(x)

    return y



print(f(-5))