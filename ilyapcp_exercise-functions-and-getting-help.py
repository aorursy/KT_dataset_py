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

    return round(num, 2)



help(round)



q1.check()
# Uncomment the following for a hint

q1.hint()
# Or uncomment the following to peek at the solution

q1.solution()
# Put your test code here

import math as math



num = 1375628345

meaningfulDigit = 4 - math.ceil(math.log10(num))

print( f"number = {num}" )

print( f"meaningful value of number = {round(num, meaningfulDigit):,d}" )
q2.check()
q2.solution()
def to_smash(total_candies, friends_num=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between the given number of friends (3 is default).

    

    >>> to_smash(91)

    1

    """

    return total_candies % friends_num



q3.check()
q3.hint()
q3.solution()
round_to_two_places(9.9999)
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x), abs(y))
def f(x):

    y = abs(x)

    return y



print(f(5))
def smallest_stringy_number(s1, s2, s3):

    return min(s1, s2, s3, key=int)
q5.check() 

q5.solution() #for q6, q7, q8.a and q8.b
q6.check() 

q6.solution() #for q7, q8.a and q8.b
q7.check() 

q7.solution() #for q8.a and q8.b
q8.a.check() 

q8.a.solution() #for q8.b
q8.b.check() 

q8.b.solution() 