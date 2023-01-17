def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    num= round(num,2)

    return num

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass

q1.check()
# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    num= round(num,2)

    return num

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass

q1.check()
# Uncomment the following for a hint

q1.hint()

# Or uncomment the following to peek at the solution

q1.solution()
# Put your test code here

help(round)

# Check your answer (Run this code cell to receive credit!)

q2.solution()
def to_smash(total_candies,total_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(10,3)

    1

    """

    

    return total_candies % total_friends



# Check your answer

q3.check()
q3.hint()
q3.solution()
# ruound_to_two_places(9.9999)

round(9.9999, 2)
x = -10

y = 5

print("Which of the two variables above has the smallest absolute value?")

smallest_abs = min(abs(x),abs(y))

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(5))