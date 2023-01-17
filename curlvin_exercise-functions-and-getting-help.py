# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    # Replace this body with your own code.

    return round(num,2)

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)





q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

round(3.14678,-2)
#q2.solution()
def to_smash(total_candies,friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    

    >>> to_smash(10,2)

    1

    """

    return (total_candies % friends)

to_smash(10,2)



q3.check()
#q3.hint()
#q3.solution()
round(9.9999,2)
x = abs(-10)

y = abs(5)

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(x, y)

print(smallest_abs)
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

    t_1=time()

#     x=fn(arg)

    t_2=time()

    return 'it took %s seconds to return %s as the answer.'%(t_2-t_1,x)

time_call(sum,range(10))

    

    

    

    

    

   



#q5.hint()

#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):

    """Return the amount of time taken by the slowest of the following function

    calls: fn(arg1), fn(arg2), fn(arg3)

    """

    t0=time()

    ans1=fn(arg1)

    ans1_txt='fn(%s)'%(arg1)

    t1=time()

    T1=t1-t0 

    

    t2=time()

    ans2=fn(arg2)

    ans2_txt='fn(%s)'%(arg2)

    t3=time()

    T2=t3-t2

    

    t4=time()

    ans3=fn(arg3)

    ans3_txt='fn(%s)'%(arg3)

    t5=time()

    T3=t5-t4

    

    answer_txt=[ans1_txt,ans2_txt,ans3_txt]

    answers=[ans1,ans2,ans3]

    times=[T1,T2,T3]

    result=zip(answer_txt,answers,times)

    

    for fnc,answer,timer in result:

        if time==max(times):

            break

    print( ' the slowest function is %s because it took %s seconds to give an answer of %s.'%(fnc,answer,timer))



slowest_call(sum,range(100),range(500),range(1000))

    

    
#q6.hint()
#q6.solution()