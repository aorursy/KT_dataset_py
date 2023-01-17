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

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

# q1.solution()
# Put your test code here

def round_to_sth(numero):

    print (round(numero, ndigits=-1))

    print (round(numero, ndigits=-2))

    print (round(numero, ndigits=-3))

    print (round(numero, ndigits=-4))



number = 14556.12145

round_to_sth(number)
# q2.solution()
def to_smash(total_candies, friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between friends.

    

    >>> to_smash(91, friends=3)

    1

    """

    return total_candies % friends



q3.check()
# q3.hint()
# q3.solution()
def round_to_two_places(num = 9.9999):

    return round(num, 2)



round_to_two_places()
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(x, y, key = abs)

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(5))