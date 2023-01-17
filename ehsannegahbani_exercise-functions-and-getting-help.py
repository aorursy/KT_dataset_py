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
    # Replace this body with your own code.
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    temp1 = num*100
    temp2 = round(temp1)
    return temp2/100

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
round(2344.56, -1)
q2.solution()
def to_smash(total_candies,n=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between n (default 3) friends.
    
    
    
    >>> to_smash(91)
    1
    """
    return total_candies % n

q3.check()
#q3.hint()
#q3.solution()

def ruound_to_two_places(n=9.9999):
    
    return round(n,3)
ruound_to_two_places(n=3.8199)
    
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(x, y, key=abs)
print("smallest abs between -10 and 5 is:", smallest_abs)
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
def sleep_fn(duration = 12):
    sleep(duration)
    pass


def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1 = time()
    sleep_fn(arg)
    t2 = time()
    print("It took", t2-t1, "sec to call the function")
    
    pass

time_call(sleep_fn, 13)
q5.hint()
q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    t1 = time_call(sleep_fn, arg1)
    t2 = time_call(sleep_fn, arg2)
    t3 = time_call(sleep_fn, arg3)
    return max(t1, t2, t3)
    pass
q6.hint()
q6.solution()
print(print("Spam"))
x = print("spam")
print(x)
q7.hint()
# help(0)
# Uncomment for an explanation.
#q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('011', '12', '3')
q8.a.solution()
help(int)
# help(int)
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3, key=int)

q8.b.check()
q8.b.hint()
#q8.b.solution()