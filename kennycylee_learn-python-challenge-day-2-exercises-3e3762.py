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
    return round(num,2)

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
q2.solution()
round(3423.324,-1)
def to_smash(total_candies,total_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % total_friends

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
##Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x),abs(y))
if smallest_abs == abs(x):
    print(x)
else:
    print(y)
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
    pass
    t0 = time()
    fn(arg)
    t1 = time()
    elapsed = t1 - t0
    return elapsed

time_call(sleep,2)
q5.hint()
q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    pass
##    t0 = time()
##    fn(arg1)
##    t1 = time()
##    fn(arg2)
##    t2 = time()
##    fn(arg3)
##    t3 = time()
##    return max(t3-t2,t2-t1,t1-t0)
    return max(time_call(fn, arg1), time_call(fn, arg2), time_call(fn, arg3))
    ##Laziness is one of the three great virtues of a programmer

slowest_call(sleep,1,3,5)
#q6.hint()
q6.solution()
print(print("Spam"))
##Solution: If you've tried running the code, you've seen that it prints:

##Spam
##None
##What's going on here? The inner call to the print function prints the string "Spam" of course. The outer call prints the value returned by the print function - which we've seen is None.

##Why do they print in this order? Python evaluates the arguments to a function before running the function itself. This means that nested function calls are evaluated from the inside out. Python needs to run print("Spam") before it knows what value to pass to the outer print.
#q7.hint()
# Uncomment for an explanation.
q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('1', '2', '3')

##Solution: smallest_stringy_number('10', '2', '3') is one example of a failure - it evaluates to '10' rather than '2'.
##The problem is that when min is applied to strings, Python returns the earliest one in lexicographic order (i.e. something like the logic used to order dictionaries or phonebooks) rather than considering their numerical value.
q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    list_of_numbers = [s1, s2, s3]
    return min(list_of_numbers, key=int)
    

q8.b.check()
#q8.b.hint()
#q8.b.solution()