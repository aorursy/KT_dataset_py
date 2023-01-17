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
    num = round(num, 2)
    return num
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
def to_smash(total_candies, num_of_division = 3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies distributed between num_of_division friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % num_of_division

q3.check()
#q3.hint()
#q3.solution()
# ruound_to_two_places(9.9999)
round(9.9999, 2)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), abs(y))
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
while duration > 0:
    print(duration, "sec")
    sleep(1)
    duration -= 1
sleep(duration)  # we can remove it
print("I'm back. What did I miss?")
def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    first = time()          # real time now in seconds
    fn(arg)                 # now applying given Func and value
    now = time()            # time after processing it
    passed = now - first    # the passed time between this period
    return passed

time_call(sleep, 3.5)
q5.hint()
q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
#------------------------------
    def time_call(fn, arg1):
        first = time()          # real time now in seconds
        fn(arg1)                 # now applying given Func and value
        now = time()            # time after processing it
        passed = now - first    # the passed time between this period
        return passed
#------------------------------
    def time_call(fn, arg2):
        first = time()          # real time now in seconds
        fn(arg2)                 # now applying given Func and value
        now = time()            # time after processing it
        passed = now - first    # the passed time between this period
        return passed
#------------------------------
    def time_call(fn, arg3):
        first = time()          # real time now in seconds
        fn(arg3)                 # now applying given Func and value
        now = time()            # time after processing it
        passed = now - first    # the passed time between this period
        return passed
#------------------------------
    return max(time_call(fn, arg1), time_call(fn, arg2), time_call(fn, arg3))
#------------------------------

slowest_call(sleep, 1, 3, 2)

q6.hint()
q6.solution()
print(print("Spam"))
#q7.hint()
#Uncomment for an explanation.
#q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number("a", "A", "999")

q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return str(min(int(s1), int(s2), int(s3)))

smallest_stringy_number("10","2","3")

q8.b.check()
#q8.b.hint()
#q8.b.solution()