# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')
def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    # Replace this body with your own code.
    return round(num,2)

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
test_number = 134.56789
print(round(test_number,-2))
q2.solution()
def to_smash(total_candies,total_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between the given number of friends.if no number
    of friends is given, the function will assume 3 is the number of friends
    
    >>> to_smash(91,3)
    1
    """
    return total_candies % total_friends

q3.check()
q3.hint()
#q3.solution()
#There was a mispelling error in calling the function
round_to_two_places(9.9999)
#help(abs)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
#As abd function only receives one parameter, it was taking the 2 values as 2 parameters at the same time generating an error
smallest_abs = min(abs(x),abs(y))
print(smallest_abs)
def f(x):
    y = abs(x)
    return y

print(f(5))
#The error here was a problem with identation an Python is very strict on that matter
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
    t0 = time()
    fn(arg)
    t1 = time()
    elapsed = t1 - t0
    return elapsed
time_call(sleep, 2)
q5.hint()
q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    a = time_call(fn,arg1)
    b = time_call(fn,arg2)
    c = time_call(fn,arg3)
    slowest = min(a,b,c)
    return slowest

slowest_call(sleep,4,6,1)
q6.hint()
#q6.solution()
print(print("Spam"))
#q7.hint()
# Uncomment for an explanation.
q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('1', '2' , '3')
#In order the function works all the elements you enter need to have the same type. otherwise, it will raise a type error
#smallest_stringy_number('1', 2 , '3') # like here where we have 2 strings and one integer in the middle
q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3, key=int)

q8.b.check()
q8.b.hint()
q8.b.solution()