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
    return round(num, 2)
    pass

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
q1.solution()
#q2.solution()
def to_smash(total_candies, number_of_friends = 3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between n number friends.
    if number of friends not provided default value is 3
    
    >>> to_smash(91)
    1
    """
    return total_candies % number_of_friends

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x),abs(y))
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
from time import sleep
duration = 5
print("Getting sleepy. See you in", duration, "seconds")
sleep(duration)
print("I'm back. What did I miss?")
from time import time, sleep
def time_call(fn, arg):
    start_time = time()
    fn(arg)
    end_time = time()
    return end_time - start_time

def sl(n):
    sleep(n)

print(time_call(sl, 5))
#q5.hint()
#q5.solution()
from time import time
def slowest_call(fn, arg1, arg2, arg3):
    exec_time = (arg1, arg2, arg3)
    max_time = 0
    for k in exec_time:
        t = time()
        fn(k)
        e = time() -t
        if(e > max_time):
            max_time = e
    return max_time

def sl(n):
    sleep(n)

print(slowest_call(sl, 1, 2, 3))
    
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
    s1 = int(s1, 10)
    s2 = int(s2, 10)
    s3 = int(s3, 10)
    return str(min(s1, s2, s3))

smallest_stringy_number('1', '2', '3')
q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    s1 = int(s1, 10)
    s2 = int(s2, 10)
    s3 = int(s3, 10)
    return str(min(s1, s2, s3))

q8.b.check()
#q8.b.hint()
#q8.b.solution()