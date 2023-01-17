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
    
    #help(round)   
    return round(num, 2)

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
#help(round)
print(round(3.141, ndigits=0))
print(round(3.141))
print(round(3.141, ndigits=1))
print(round(3.141, ndigits=-1))
print(round(32.141, ndigits=-1))
print(round(325.1141, ndigits=-1))
print(round(325.1141, ndigits=1))
print(round(325.1141, ndigits=-2))
print(round(325.1141, ndigits=2))
print(round(45325.1141, ndigits=-2))
#q2.solution()
def to_smash(total_candies, number_of_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between number_of_friends friends.
    Default value for number_of_friends is 3
    
    >>> to_smash(91)
    1
    """
    return total_candies % number_of_friends

q3.check()
#q3.hint()
#q3.solution()
# suspect wrong function name
# ruound_to_two_places(9.9999)

# Corrected below
round_to_two_places(9.9999)
# suspect function call is wrong..lets uncomment and see :)
x = -10
y = 5
# # Which of the two variables above has the smallest absolute value?
# smallest_abs = min(abs(x, y))

# also noticed passing 2 arguments to abs() which is wrong
# Corrected below
smallest_abs = abs(min(x, y))
print('Smallest absolute value b/w ', x ,' and ', y, ' is ', smallest_abs)
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
def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t_start = time()
    fn(arg)
    t_end = time()
    return (t_end - t_start)
    #time_taken = t_end - t_start
    #print('Time Taken : ', time_taken)

    
# testing time_call function
time_call(sleep, 4)
#q5.hint()
#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    return max(fn(sleep, arg1), fn(sleep, arg2), fn(sleep, arg3))
    #return max(time_call(fn, arg1), time_call(fn, arg2), time_call(fn, arg3))

# testing slowest_call function 
# slowest_call(sleep, 2, 1, 3)
slowest_call(time_call, 2, 1, 3)
#q6.hint()
#q6.solution()
# my predection: no error; inner print will print Spam ; not sure on how outer print will act.. uncomments and check.. :)
#print(print("Spam"))
print(print(print("Spam")))
#q7.hint()
# Uncomment for an explanation.
#q7.solution()
# My predection: no error; it returns '1' consider as string
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

# smallest_stringy_number('1', '2', '3') # Output '1'
# smallest_stringy_number('One', '2', '3') # Output '2'
# smallest_stringy_number('One', 'Two', 'Three') # Output 'One' as per alphabetical order 
# smallest_stringy_number('2', '11', '3') # Output 11 
smallest_stringy_number('One', 'Four', 'Three') # Output Four

#q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3, key=int)

q8.b.check()
#q8.b.hint()
#q8.b.solution()