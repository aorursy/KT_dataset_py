# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')

#any
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    #result= round(num,2)

    #print(result)

    return round(num,2)



    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass

#round_to_two_places(3.14159)

x=float(input("Enter any number: "))

print(round_to_two_places(x))



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

#help(round)

#print(round(3.14159,-2))



#print(round(1345.45632, -1))



print("Helpful when dealing with large numbers ")

#print(round(134545632, -3))



def round_to_places(num, rnd=2):

    

    return round(num,rnd)



    # Replace this body with your own code.

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass

round_to_places(33443.14159, -2)

#x=float(input("Enter any number: "))

#print(round_to_two_places(x))
q2.solution()
def to_smash(total_candies ,friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    return total_candies % friends

print(to_smash(91))



#user defined input

x= int(input("Enter total candies: "))

#y=int(input("Enter total griends: "))

print(to_smash(x))

q3.check()
q3.hint()
q3.solution()
print(round_to_places(9.9999))

#print("rounding off")
def f(x):

    y = abs(x)

    return y

print(f(5))