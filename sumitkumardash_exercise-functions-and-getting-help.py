# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    return(round(num,2))

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
print(round(212,-2))
print(round(251,-3))
#q2.solution()
def to_smash(total_candies,no=3):

    if no!=3 :

        return(total_candies%no)

    else:

           

        return total_candies % 3



q3.check()
#q3.hint()
#q3.solution()
def ruound_to_two_places(n):

    print(round(n))

ruound_to_two_places(9.9999)
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?



smallest_abs = min(abs(x),abs(y))

print(smallest_abs)

def f(x):

    y = abs(x)

    return y



print(f(5))