# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    return round(num, 2)

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

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

# help(round)

print(round(2334.414, -2))

print(round(15.454, -1))

print(round(199.222, -1))
#q2.solution()
def to_smash(total_candies, friends=3):

    

    return total_candies % friends



q3.check()
#q3.hint()
#q3.solution()
def round_to_two_places(num):

    return round(num, 2)

round_to_two_places(99.999)

round_to_two_places(93.31119)

round_to_two_places(645.35623)

round_to_two_places(0.4459)



x = -10

y = 5

# # Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x), abs(y))

smallest_abs
def f(x):

    y = abs(x)

    return y



print(f(5))