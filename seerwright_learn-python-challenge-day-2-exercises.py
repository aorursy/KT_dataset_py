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
    return round(num, 2)

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
#q2.solution()
def to_smash(total_candies, friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between n friends.
    
    @param total_candies: [required] Number of candies to be distributed
    @param friends: [optional] Number of friends to distribute candies among, default is 3
    @return: Number of candies that must be smashed so that the friends can have equal numbers of candies
    """
    return total_candies % friends

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min([abs(i) for i in [x, y]])
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
from time import time
from random import randint

def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1 = time()
    fn(arg)
    t2 = time()
    #print((t2 - t1), "seconds since the Epoch")
    return (t2 - t1)
    
def some_function(n=5):
    '''
    Silly function to wait a few seconds. Prints a little note each second.
    @param n: [optional] Number of seconds to wait in total. Defaults to 5. Must be in range 1..5
    @return: Nothing. This is a side-effect function
    '''
    if n < 1 or n > 5:
        n = 5
    wait_seconds = randint(1, n)
    for s in range(1, n+1):
        msg = 'Waiting ...' if s == 1 else 'Still waiting ...'
        sleep(1)
        print(msg)

time_call(some_function, 4)
#q5.hint()
#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    times = []
    for arg in [arg1, arg2, arg3]:
        print('\nChecking argument %d' % arg)
        times.append(time_call(fn, arg))
    
    return min(times)

slowest_call(some_function, 2, 3, 4)
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

# Double digits is kind of goofy since it evaluates on first, then second, etc. Not the number as a whole
#smallest_stringy_number('1', '2', '03')
# This will fail because of mixed types
#smallest_stringy_number(0, '2', '3')
# ASCII order may not makes sense here
#smallest_stringy_number('A', 'a', '_')
# q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    
    # Two options here:
    # List comprehension
    #return str(min([int(i) for i in [s1, s2, s3]]))
    # Lambda function
    return min([s1, s2, s3], key=lambda i: int(i))

q8.b.check()
#q8.b.hint()
#q8.b.solution()