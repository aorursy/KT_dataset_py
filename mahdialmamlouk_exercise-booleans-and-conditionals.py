from learntools.core import binder; binder.bind(globals())

from learntools.python.ex3 import *

print('Setup complete.')
# Your code goes here. Define a function called 'sign'

def sign(a):

    if a == 0:

        return 0

    elif a > 0:

        return 1

    else:

        return -1



q1.check()
#q1.solution()

q1.solution()
def to_smash(total_candies):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends.

    

    >>> to_smash(91)

    1

    """

    if total_candies <= 1:

        print ("Splitting", total_candies, "candy")

    else:

        print("Splitting", total_candies, "candies")

    return total_candies % 3



to_smash(31)

to_smash (1)

to_smash (0)
to_smash(1)
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



q3.check()
#q3.hint()

#q3.solution()

q3.hint()

q3.solution()
def is_negative(number):

    if number < 0:

        return True

    else:

        return False



def concise_is_negative(number):

    pass # Your code goes here (try to keep it to one line!)



q4.check()
#q4.hint()

#q4.solution()

q4.solution()
def onionless(ketchup, mustard, onion):

    """Return whether the customer doesn't want onions.

    """

    return not onion
def wants_all_toppings(ketchup, mustard, onion):

    """Return whether the customer wants "the works" (all 3 toppings)

    """

    return ketchup and mustard and onion



q5.a.check()
#q5.a.hint()

#q5.a.solution()

q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):

    """Return whether the customer wants a plain hot dog with no toppings.

    """

    return not ketchup and not mustard and not onion



# Check your answer

q5.b.check()
#q5.b.hint()

#q5.b.solution()

q5.b.solution()
def wants_plain_hotdog(ketchup, mustard, onion):

    """Return whether the customer wants a plain hot dog with no toppings.

    """

    return (ketchup+mustard)==1

q5.b.check()
#q5.c.hint()

#q5.c.solution()

q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):

    """Return whether the customer wants exactly one of the three available toppings

    on their hot dog.

    """

    return (ketchup+mustard+onion)==1



q6.check()
#q6.hint()

#q6.solution()

q6.solution()

def should_hit(player_total, dealer_total, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    return ((player_total<=14)

            #or( player_aces==1) 

            #or (player_aces==1)

            #or (player_total<= dealer_total )

           )

       
q7.simulate_one_game()
q7.simulate(n_games=50000)
def should_hit(player_total, dealer_total, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    return  player_total<=14



q7.simulate(n_games=1000)