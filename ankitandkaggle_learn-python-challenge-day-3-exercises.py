# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
# Your code goes here. Define a function called 'sign'
def sign(num):
    if num < 0:
        return -1
    elif num ==0:
        return 0
    elif num >0:
        return 1
print(sign(-1))
print(sign(2))
print(sign(0))
print(sign(-0)) # just to check, -0 is treated as 0 probably because there is nothing like -0
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
    if total_candies==1:
        print("Splitting", total_candies, "candy")
    else:
        print("Splitting", total_candies,"candies")
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
    if ketchup and mustard:
        return False
    if (not ketchup and not mustard):
        return False
    if (not ketchup and mustard) or onion:
        return True
    if (ketchup and not mustard) or onion:
        return True
    
    

q5.c.check()
#using definition of xor gate (a and (not b)) or ((not a) and b)
def exactly_one_sauce(ketchup, mustard, onion):
    return (ketchup and not mustard) or (not ketchup and mustard)
q5.c.check()
#q5.c.hint()
#q5.c.solution()
print(int(True))
print(int(5 == 0)) #5==0 returns false so final output is 0
def exactly_one_topping(ketchup, mustard, onion):
    if int(ketchup) + int(mustard) + int(onion) == 1:
        return True
    else:
        return False

q6.check()
#q6.hint()
#q6.solution()
def should_hit(player_total, dealer_total, player_aces):
    if player_total<dealer_total:
        return True
   
q7.simulate_one_game()
q7.simulate(n_games=1000)
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if (player_total<=17 and player_total<dealer_total):
        return True
    
q7.simulate(n_games=100000)

