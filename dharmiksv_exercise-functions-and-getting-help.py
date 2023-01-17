# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
class check:

        def __init__(self,num):

            self.num = num

        

        def check(self):

            return round(self.num,2)

        

q1 = check(9.9999)



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
class check:

        def __init__(self,num,ndigits):

            self.num = num

            self.ndigits = ndigits

        

        def check(self):

            return round(self.num,self.ndigits)

        

q2 = check(3.14159,-1)

q2.check()
class check:

        def __init__(self,num,ndigits):

            self.num = num

            self.ndigits = ndigits

        

        def check(self):

            return round(self.num,self.ndigits)

        

q1 = check(3.14159,-2)



q2.check()


class check:

        def __init__(self,total_candies,total_friends):

            self.total_candies = total_candies

            self.total_friends = total_friends

                        

        

        def check(self):

            return self.total_candies % self.total_friends

        

q3 = check(91,3)

q3.check()
#q3.hint()
#q3.solution()
def ruound_to_two_places(num):

    return round(num,1)



ruound_to_two_places(9.9999)
x = -10

y = 5

# # Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x),abs(y))

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(5))