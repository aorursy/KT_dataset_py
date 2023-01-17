# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')
def round_to_two_places(num):
    return round(num ,2)
    round_to_two_places(40.58)
    pass
# Check your answer
q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
q1.solution()
# Put your test code here

def round_to_two_places(num):
    return round(num ,-2)
round_to_two_places(40.89)
q2.check()
# Check your answer (Run this code cell to receive credit!)
q2.solution()
def to_smash(total_candies, n_friends=3):
    return total_candies % n_friends
    to_smash(121, 3)
# Check your answer
q3.check()
q3.hint()
q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# # Which of the two variables above has the smallest absolute value?
smallest_abs = abs(max(x,y))
smallest_abs
def f(x):
    y = abs(x)
    return y

print(f(-5))