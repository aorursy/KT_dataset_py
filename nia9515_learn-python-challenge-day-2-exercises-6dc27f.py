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
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    pass

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
#q2.solution()
def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % 3

q3.check()
#q3.hint()
#q3.solution()
# ruound_to_two_places(9.9999)
# x = -10
# y = 5
# # Which of the two variables above has the smallest absolute value?
# smallest_abs = min(abs(x, y))
# def f(x):
#     y = abs(x)
# return y

# print(f(5))
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
x=time()
def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    return fn(arg)
y = time ()
z=y-x
print(z)
a1=time()
a=time_call(sleep,5)
b1=time()
c=b1-a1
print(c)
#y= time_call(1,2)
#print(y)
#z=y-x
#print(z)
#check statement



    
q5.hint()
q5.solution()
from time import time 
from time import sleep
def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    return fn(arg)
a=time_call(sleep,1)
print(a)

def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    return max(time_call(fn,arg1),time_call(fn,arg2),time_call(fn,arg3))

slowest_call(time_call,1,2,3)


q6.hint()
q6.solution()
print(print("Spam"))
q7.hint()
# Uncomment for an explanation.
q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('s', 's', 's')
q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    s1=int(s1)
    s2=int(s2)
    s3=int(s3)
    return str(min(s1, s2, s3))

q8.b.check()
q8.b.hint()
q8.b.solution()