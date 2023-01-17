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
    #pass -- commented out 'pass'
    return round(num, 2)

q1.check()
#help(round)
print(round(3.14159, 2))
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
help(abs)
#q2.solution()
#def to_smash(total_candies):
def to_smash(total_candies, num_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between num_friends (default 3) friends.
    
    >>> to_smash(91, 3)
    1
    """
    return total_candies % num_friends

print(to_smash(91, 4))

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), abs(y))
print("smallest_abs:", smallest_abs)
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
    from time import time
    # get pre_fn_time
    pre_fn_time = time()
    # call fn
    fn(arg)
    # get post_fn_time
    post_fn_time = time()
    # calc
    fn_time = post_fn_time - pre_fn_time
    return fn_time

#q5.hint()
time_call(sleep, 5)
q5.solution()
# write a function that calls sleep for varying amounts of time

def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    #pass
    arg1_time = time_call(fn, arg1)
    arg2_time = time_call(fn, arg2)
    arg3_time = time_call(fn, arg3)
    return max(arg1_time, arg2_time, arg3_time)
# I'd use something like this for performance testing but pass the same arg to different functions that perform the same task.
print("slowest_call of sleep for 4, 8, 6 is:", slowest_call(sleep, 4, 8, 6) )
#q6.hint()
#q6.solution() # I'm comfortable with my slightly less efficient alternative.
print(print("Spam"))
q7.hint()
help(abs(-2))
# Uncomment for an explanation.
q7.solution()
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
    #return str(min(int(s1), int(s2), int(s3)))
    #the following is better than my solution, also judged correct
    return min(s1, s2, s3, key=int)

q8.b.check()
q8.b.hint()
help(min)
#q8.b.solution()