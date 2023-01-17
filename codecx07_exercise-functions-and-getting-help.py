# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')
def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    return (round(num,2))


q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
#q2.solution()
def to_smash(total_candies, no_of_friends = 3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between n friends.
    
    >>> to_smash(91,4)
    3
    """
    return total_candies % no_of_friends

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x),abs(y))
def f(x):
    y = abs(x)
    return y

print(f(5))
# Importing the function 'time' from the module of the same name. 
# (We'll discuss imports in more depth later)
from time import time
t = time()
print(t, "seconds since the Epoch")
from time import sleep
duration = 5
print("Getting sleepy. See you in", duration, "seconds")
sleep(duration)
print("I'm back. What did I miss?")
def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    return(time())

amount_of_time = -time() + time_call(1, 2)
print(amount_of_time)
q5.hint()
q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    elapsed1i = time()
    fn(arg1)
    elapsed1o = time()
    elapsed1 = elapsed1o - elapsed1i
    elapsed2i = time()
    fn(arg2)
    elapsed2o = time()
    elapsed2 = elapsed2o - elapsed2i
    elapsed3i = time()
    fn(arg3)
    elapsed3o = time()
    elapsed3 = elapsed3o - elapsed3i
    return (max(elapsed1, elapsed2, elapsed3))
#q6.hint()
q6.solution()
print(print("Spam"))
#q7.hint()
# Uncomment for an explanation.
#q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('1', '2', '3')
q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3, key = int)

q8.b.check()
#q8.b.hint()
q8.b.solution()
