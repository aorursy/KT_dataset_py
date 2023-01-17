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
    return round(num,2)

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
print(round(2.3,-1))
print(round(2.333,-1))
print(round(23.3,-1))
print(round(2333.3,-2))
print(round(20211,-3))
print(round(222.3,-2))
print(round(20.3,-2))
print(round(25.3,-3))
#as it rounds the number or decimal to its nearest tens, hundreds or thousands
#it can be used to get the min range or to just have the rouded value of big numbers
#for example if a country or city's population is 12000433 we can just used this round(12000433,-3)
#and says that country's population is more than 12million
print(round(12000433,-3))
#q2.solution()
def to_smash(total_candies,number_of_friends = 3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % number_of_friends
print(to_smash(223)) #calling it using just one argument and rest if the default arg as in definition
print(to_smash(233,5)) #this replaces the number_of_friends value from 3 to 5
q3.check()
#q3.hint()
#q3.solution()
# ruound_to_two_places(9.9999)
#there is smelling mistake in above line
round_to_two_places(9.9999)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), abs(y))
smallest_abs
def f(x):
    y = abs(x)
#return y  #incorrect indentation
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
import math
def time_call(fn, arg):
    t1 = time()
    fn(arg)
    t2 = time()
    print(t2-t1)
time_call(math.sqrt,4)

#q5.hint()
#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    time1 = time_call(fn,arg1)
    time2 = time_call(fn,arg2)
    time3 = time_call(fn,arg3)
    return max(time1,time2,time3)
#q6.hint()
#q6.solution()
print(print("Spam")) #as python goes from inside to outside, it will first print Spam using print functon inside paranthesis
# as print return None so now it will print None which was returned from the print inside parenthesis
#q7.hint()
# Uncomment for an explanation.
#q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('1', '2', '3')
#q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3,key = int)

q8.b.check()
q8.b.hint()
#q8.b.solution()