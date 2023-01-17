# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
# Your code goes here. Define a function called 'sign'
def sign(number):
    # Two ternary ops instead of if/elif/else
    sign = -1 if number < 0 else 1
    sign = 0 if number == 0 else sign
    return sign
    
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
    return total_candies % 3

to_smash(91)
to_smash(1)
#q2.hint()
#q2.solution()
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
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

def concise_is_negative(number):
    return True if number < 0 else False

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
    return not (ketchup or mustard or onion)

q5.b.check()
#q5.b.hint()
#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    def xor(a, b):
        return (a or b) and (not(a and b))
    
    return(xor(ketchup, mustard))

q5.c.check()
#q5.c.hint()
#q5.c.solution()
import math
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    # Get args as bits
    args = locals()
    bits = [1 if args[topping]==True else 0 for topping in args]
    # Figure out base 10 version of toppings
    bits.reverse()
    base_10 = sum([(2 ** position)*b for position,b in enumerate(bits)])
    # Is a 3-way XOR if base_10 goes evenly back to base_2 
    # (Monkeymod on the else, but we don't calculate on actual base_2)
    base_2 = math.log(base_10, 2) if base_10 > 0 else 0.0000001
    is_square = True if base_2 % 1 == 0 else False
    
    return is_square

q6.check()
#q6.hint()
#q6.solution()
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    hit_me = False
    if player_total < 17:
        hit_me = True
    
    return hit_me
q7.simulate_one_game()
q7.simulate(n_games=1000)
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    # I want to see the actual cards!!
    hit_me = False
    if dealer_total >= 19:
        hit_me = True
    elif dealer_total >= 18 and dealer_total < 19:
        hit_me = True
    elif player_total < 17:
        hit_me = True
        
    return hit_me

q7.simulate(n_games=1000)