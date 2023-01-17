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

    return round(num, 2)

    

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    

    

    # because after we begin a code block, Python requires at least one line of code)

    



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

def round2(num, n = -3): #rounds to nearest 1000 as default

    return round(num,-4) # rounding to nearest 10000



#q2.solution()
def to_smash(total_candies, n=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % n





q3.check()













#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10

y = 5

 # Which of the two variables above has the smallest absolute value?

x,y = abs(x) , abs(y)

smallest_abs = min(x, y)

print(smallest_abs)
def f(x):

    

    y = abs(x)

    

    return y



print(f(5))