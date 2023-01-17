# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
help(round)
def round_to_two_places(num):

    return round(num,2)

    

round_to_two_places(3.14159)
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    return round(num, 2)



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

# round to tens, hundreds, thousands ,etc, the larger the negative number

# ndigits=-1 rounds to the nearest 10, ndigits=-2 rounds to the nearest 100 and so on. 

print(round(16618, -1))
#q2.solution()
def to_smash(total_candies, friends = 3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    Defaults is three

    >>> to_smash(91)

    1

    """

    return total_candies % friends







q3.check()
#testing function above

print(to_smash(10))

print(to_smash(10, friends = 2))
#q3.hint()
#q3.solution()
#rounds to two decimL Places 



def round_to_two_places(x):

    return round(x,2)

round_to_two_places(9.9999)
x = -10

y = 5



# Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x), abs(y))



print(smallest_abs)
#print absolute value of input 



def f(x):

     y = abs(x)

     return y



print(f(5))

print(f(-5))