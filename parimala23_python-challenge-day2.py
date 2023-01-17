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
    num1=round(num,2)
    return num1
    
    pass

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
a=12.34546
b=round(a,-1)
print(b)
#q2.solution()
def to_smash(total_candies,n_friends=3):
    return total_candies%n_friends

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# # Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), y)
print(smallest_abs)
def f(x):
     y = abs(x)
     return y

print(f(5))
# Importing the function 'time' from the module of the same name. 
# (We'll discuss imports in more depth later)
from time import time
t = time()
print(t, "seconds since the Epoch")
import time
duration = 5
print("Getting sleepy. See you in", duration, "seconds")
sleep(duration)
print("I'm back. What did I miss?")
from time import sleep 
def f(n):
    return n*2
def time_call(fn,arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1=time()
    fn(arg)
    t2=time()
    elapsed=t12-t1
    print(elapsed)
print(time_call(f,2))
   
    



   

q5.hint()
q5.solution()
import time 
def f(n):
    return n*2
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    t1=time()
    fn(arg1)
    t2=time()
    targ1=t12-t1
    t1=time()
    fn(arg2)
    t2=time()
    targ2=t12-t1
    t1=time()
    fn(arg3)
    t2=time()
    targ3=t12-t1
    return min(targ1,targ2,targ3)    
    
    print(slowest_call(f, 2, 3, 4))
#q6.hint()
#q6.solution()
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

smallest_stringy_number(1, 2, 3)
#q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)
smallest_stringy_number(1, 2, 3)
#q8.b.check()
#q8.b.hint()
#q8.b.solution()