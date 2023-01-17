from learntools.core import binder; binder.bind(globals())

from learntools.python.ex3 import *

print('Setup complete.')
# Your code goes here. Define a function called 'sign'

def sign(x):

    if x > 0:

        return 1

    elif x < 0:

        return -1

    else:

        return 0

# Check your answer

q1.check()
#q1.solution()
def to_smash(total_can):

    if total_can == 1:

        print("splitting 1 candy")

    else:

        print("splitting", total_can, "candies")

    return total_can % 3



to_smash(91)
to_smash(1)
def to_smash(total_can):

    print("splitting", total_can, "candi" if total_can == 1 else "candie")

    return total_can % 3



to_smash(91)

to_smash(1)
# Check your answer (Run this code cell to receive credit!)

#q2.solution()
have_umbrella = True

rain_level = 0.0

have_hood = True

is_workday = True

def red_for_weather(have_umbrella, rain_level, have_hood, is_workday):

    return (not (rain_level > 0)) and is_workday



actual = red_for_weather(have_umbrella, rain_level, have_hood, is_workday)

print(actual)



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
def is_negative(num):

    if num < 0:

        return True

    else:

        return False



def concise_is_negative(num):

    return num < 0



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
def onionless(ketchup, mustard, onion):

    return not onion
def wants_all_toppings(ketchup, mustard, onion):



    return ketchup and mustard and onion





# Check your answer

q5.a.check()
#q5.a.hint()

#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):



    return not ketchup and not mustard and not onion



# Check your answer

q5.b.check()
#q5.b.hint()

#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):



    return (ketchup and not mustard) or (mustard and not ketchup)



# Check your answer

q5.c.check()
#q5.c.hint()

#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):



    return (int(ketchup) + int(mustard) + int(onion)) == 1



# Check your answer

q6.check()
#q6.hint()

#q6.solution()
import random

random.seed(0)

def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):



    return random.seed(0)
q7.simulate_one_game()
q7.simulate(n_games=50000)
import random

random.seed(0)

def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):



    return random.seed(0)



q7.simulate(n_games=50000)