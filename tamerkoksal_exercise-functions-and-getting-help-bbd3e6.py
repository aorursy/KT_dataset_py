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
    return round(num, ndigits=2)

q1.check()
help(round)
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
# ex1
print("ex1: ", round(33, -2))

# ex2
print("ex2: ", round(333.3, -2))

# ex3
print("ex3: ", round(567.8973, -2))
#q2.solution()
def to_smash(total_candies, num_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % num_friends

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), abs(y))
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
    t_start = time()
    fn(arg)
    t_finish = time()
    return t_finish - t_start

def countdown(n):
    for i in range(n):
        pass # do nothing for n times
    

print("Executing the function given the argument: ", 10**6, " has taken ", time_call(countdown, 10**6), " seconds")
print("Executing the function given the argument: ", 10**7, " has taken ", time_call(countdown, 10**7), " seconds")
print("Executing the function given the argument: ", 10**8, " has taken ", time_call(countdown, 10**8), " seconds")
#q5.hint()
#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    return max(time_call(countdown, arg1), time_call(countdown, arg2), time_call(countdown, arg3))

print("The amount of time taken by the slowest of the function calls: countdown(10**6), countdown(10**7), countdown(10**8) is ", slowest_call(countdown, 10**6, 10**7, 10**8))
#q6.hint()
#q6.solution()
print(print("Spam"))
print(Spam)
q7.hint()
# Uncomment for an explanation.
q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

print(smallest_stringy_number('1', '2', '3'))
print(smallest_stringy_number('11', '9', '8'))
#q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return str(min(int(s1), int(s2), int(s3)))

q8.b.check()
#q8.b.hint()
#q8.b.solution()