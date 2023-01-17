# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')
a = 4.12345
print(a)
b = str(a)
print(b)
def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    num = str(num)[:4]
    return float(num)
q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
#q2.solution()
people = [1,2,3,4,11,14,15,16,17,23,25,31,35,48]
ranges = list(map(lambda x: round(x , -1), people))
kids = len(list(filter((lambda x: x<=10), ranges)))
teens_youngAdults = len(list(filter((lambda x: (x>10) and (x<=20)), ranges)))
grown_ups = len(list(filter((lambda x: x>20), ranges)))
print("""
this type of rounding with negative can be used to create groups, for example: 
list of ages for some people:
[1,2,3,4,11,14,15,16,17,23,25,31,35,48]
grouped according to age:
{} kids
{} teens os young Adults
{} grown ups
""".format(kids, teens_youngAdults, grown_ups))
def to_smash(total_candies, friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between any number of friends, in case you don't give
    the number of friends, it'll be assumed you are 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % friends

q3.check()
#q3.hint()
#q3.solution()
print("prediction: python won't be able to find the function because the name is wrong")
#ruound_to_two_places(9.9999)
round_to_two_places(9.9999)
print("prediction: the abs function expects only 1 argument instead of 2")
x = -10
y = 5
# # Which of the two variables above has the smallest absolute value?
#smallest_abs = min(abs(x, y))
smallest_abs = min(abs(x),abs(y))
print (smallest_abs)
print ("prediction: error in the tabulation of the function")
# def f(x):
#     y = abs(x)
# return y
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
import math

def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    init_time = time()
    fn(arg)
    end_time = time()
    total_time = end_time - init_time
    return total_time

def wait_secs(seconds):
    sleep(seconds)
    
test_duration = 2
did_time_call_worked = True if math.floor(time_call(wait_secs, test_duration)) == test_duration else False
print ("Time_call function worked? ",did_time_call_worked)
    
#q5.hint()
#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    functions = {0:time_call(fn, arg1),1:time_call(fn, arg2),2:time_call(fn,arg3)}
    slowest_time = min(functions.values())
    return slowest_time

slowest_call((lambda x: sleep(x)),3,1,2)
#q6.hint()
#q6.solution()
print("python will evaluate that which is in parantheses first and then the outer print so it will print 'Spam' and then do nothing" )
print(print("Spam"))
#q7.hint()
# Uncomment for an explanation.
q7.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('1', '10', '021')
q8.a.solution()
def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return str(min(int(s1),int(s2), int(s3)))

q8.b.check()
#q8.b.hint()
#q8.b.solution()