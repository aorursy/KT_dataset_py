# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num): 

    return round(num,2)

    

# Check your answer

q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
help(round(123032,-2))
# Check your answer (Run this code cell to receive credit!)

q2.solution()
def to_smash(total_candies,x=3):

    return total_candies % x

# Check your answer

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10

y = 5

#Which of the two variables above has the smallest absolute value?

x=abs(x)

y=abs(y)

smallest_abs = min(x, y)

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(5))