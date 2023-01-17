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
q1.solution()
def to_smash(total_candies):
    if total_candies == 1:
        print("Splitting 1 candy")
    else:
        print ("Splitting", total_candies, "candies")
    #return total_candies % 3

to_smash(1)
to_smash(1)
def to_smash(total_candies):
    if total_candies == 1:
        print("Splitting 1 candy")
    else:
        print("Splitting", total_candies, "candies")
   # return total_candies % 3

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
    pass 
    return number < 0

# Check your answer
q4.check()
#q4.hint()
#q4.solution()
def onionless(ketchup, mustard, onion):
    return not onion
def wants_all_toppings(ketchup, mustard, onion):
    return ketchup and mustard and onion
    pass
    

# Check your answer
q5.a.check()
#q5.a.hint()
#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):
    return not ketchup and not mustard and not onion
    pass

# Check your answer
q5.b.check()
#q5.b.hint()
#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):
    return (ketchup and not mustard) or (mustard and not ketchup)
    pass

# Check your answer
q5.c.check()
#q5.c.hint()
#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    return (ketchup + mustard + onion) == 1
    pass

# Check your answer
q6.check()
#q6.hint()
#q6.solution()
def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay.
    When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
    doesn't bring the total above 21, otherwise we count them as low (with value 1). 
    For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
    and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
    """
    return False
q7.simulate_one_game()
q7.simulate(n_games=50000)
def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay.
    When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
    doesn't bring the total above 21, otherwise we count them as low (with value 1). 
    For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
    and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
    """
    return False

q7.simulate(n_games=50000)