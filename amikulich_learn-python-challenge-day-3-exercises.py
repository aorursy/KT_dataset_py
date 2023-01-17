# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0   

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
    print("Splitting", total_candies, "candy" if (total_candies % 10 == 1 and total_candies != 11) else  "candies" )
    return total_candies % 3

to_smash(91)
to_smash(1)
#q2.hint()
q2.solution()
def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):
    # Don't change this code. Our goal is just to find the bug, not fix it!
    return have_umbrella or rain_level < 5 and have_hood or not rain_level > 0 and is_workday

# Change the values of these inputs so they represent a case where prepared_for_weather
# returns the wrong answer.
have_umbrella = False
rain_level = 10.0
have_hood = False
is_workday = False

# Check what the function returns given the current values of the variables above
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

q3.check()
#q3.hint()
#q3.solution()
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

def concise_is_negative(number):
    return number < 0

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
    return ketchup and mustard and onion

q5.a.check()
#q5.a.hint()
#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):
    """Return whether the customer wants a plain hot dog with no toppings.
    """
    return not ketchup and not mustard and not onion

q5.b.check()
#q5.b.hint()
#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    return (ketchup and not mustard) or (not ketchup and mustard)

q5.c.check()
#q5.c.hint()
#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    return sum([int(ketchup), int(mustard), int(onion)]) == 1

q6.check()
#q6.hint()
#q6.solution()
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if (player_aces == 3):
        return False
    if (player_total > 18):
        return False
    if (player_total > 16 and dealer_total == 11):            
        return False    
    if (player_total > 15 and player_aces == 0):
        return False
        
    return True
q7.simulate_one_game()
q7.simulate(n_games=1000)
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if (player_aces == 3):
        return False
    if (player_total > 18):
        return False
    if (player_total > 16 and dealer_total == 11):            
        return False    
    if (player_total > 15 and player_aces == 0):
        return False

q7.simulate(n_games=1000)