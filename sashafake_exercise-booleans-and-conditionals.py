from learntools.core import binder; binder.bind(globals())

from learntools.python.ex3 import *

print('Setup complete.')
# Your code goes here. Define a function called 'sign'

def sign(num):

    if num > 0:

        return 1

    elif num == 0:

        return 0

    else:

        return -1



# Check your answer

q1.check()
#q1.solution()
def to_smash(total_candies):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    print("Splitting", total_candies, "candies")

    return total_candies % 3



to_smash(91)
to_smash(1)
def to_smash(total_candies):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    print("Splitting", total_candies, "candy" if total_candies == 1 else "candies")



to_smash(91)

to_smash(1)
# Check your answer (Run this code cell to receive credit!)

q2.solution()
def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):

    # Don't change this code. Our goal is just to find the bug, not fix it!

    return have_umbrella or rain_level < 5 and have_hood or not rain_level > 0 and is_workday



# Change the values of these inputs so they represent a case where prepared_for_weather

# returns the wrong answer.

have_umbrella = False

rain_level = 0.0

have_hood = False

is_workday = False



# Check what the function returns given the current values of the variables above

actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)

print(actual)



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
def is_negative(number):

    if number < 0:

        return True

    else:

        return False



def concise_is_negative(number):

    return True if number < 0 else False



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
def onionless(ketchup, mustard, onion):

    """Return whether the customer doesn't want onions.

    """

    return not onion
def wants_all_toppings(ketchup, mustard, onion):

    """Return whether the customer wants "the works" (all 3 toppings)

    """

    return True if ketchup and mustard and onion else False



# Check your answer

q5.a.check()
#q5.a.hint()

#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):

    """Return whether the customer wants a plain hot dog with no toppings.

    """

    return False if ketchup or mustard or onion else True



# Check your answer

q5.b.check()
#q5.b.hint()

#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):

    """Return whether the customer wants either ketchup or mustard, but not both.

    (You may be familiar with this operation under the name "exclusive or")

    """

    if ketchup and not mustard and onion:

        return True

    elif mustard and not ketchup and onion:

        return True

    elif onion and not ketchup and not mustard:

        return True

    else:

        return False



# Check your answer

q5.c.check()
#q5.c.hint()

#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):

    """Return whether the customer wants exactly one of the three available toppings

    on their hot dog.

    """

    return True if ketchup and not mustard and not onion or mustard and not ketchup and not onion or onion and not ketchup and not mustard else False



# Check your answer

q6.check()
#q6.hint()

#q6.solution()