# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

   



  return round(num,2)

    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

 



q1.check()
# Uncomment the following for a hint

q1.hint()

help(round)

# Or uncomment the following to peek at the solution

q1.solution()
# Put your test code here

number= 422546

round(number, 2)

q2.solution()
def to_smash(total_candies,number_of_friends=3):

  

    return total_candies % number_of_friends



q3.check()
#q3.hint()
q3.solution()
# ruound_to_two_places(9.9999)
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x, y))
def f(x):

    y = abs(x)



    return y



print(f(-10))