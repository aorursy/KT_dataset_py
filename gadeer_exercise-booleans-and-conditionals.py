from learntools.core import binder; binder.bind(globals())

from learntools.python.ex3 import *

print('Setup complete.')
# Your code goes here. Define a function called 'sign'

def sign(x):

    if x < 0:

      y = -1

      print('Your value', x, 'is negative')

    elif x>0:

      y = 1

      print('Your value', x, 'is positive')

    elif x == 0:

      y = 0

      print('Your value', x, 'is zero')

    else:

      y = 'No number'

      print('Your value', x, 'is not numerical')

    return y

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

    print("Splitting", total_candies, "candies")

    return total_candies % 3



to_smash(91)

to_smash(1)
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

    return number < 0



q4.check()
#q4.hint()

#q4.solution()
def onionless(ketchup, mustard, onion):

    """Return whether the customer doesn't want onions.

    """

    return not onion

onionless(ketchup=True, mustard=True, onion=False)
def wants_all_toppings(ketchup, mustard, onion):

    """Return whether the customer wants "the works" (all 3 toppings)

    """

    y = ketchup and mustard and onion

    print("wants all toppings:", ketchup and mustard and onion)

    return y



wants_all_toppings(ketchup = True, mustard = True, onion = True)

q5.a.check()

#q5.a.hint()

#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):

    """Return whether the customer wants a plain hot dog with no toppings.

    """

    y = not ketchup and not mustard and not onion

    print("wants plain hotdog:", not ketchup and not mustard and not onion)

    return y



wants_plain_hotdog(ketchup=False, mustard=False, onion=False)

wants_plain_hotdog(ketchup=False, mustard=True, onion=False)



q5.b.check()
#q5.b.hint()

#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):

    """Return whether the customer wants either ketchup or mustard, but not both.

    (You may be familiar with this operation under the name "exclusive or")

    """

    y = (ketchup ^ mustard)

    print("exactly one sauce:",  y)

    return y



exactly_one_sauce(ketchup=False, mustard=True, onion=False)

#q5.c.hint()

#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):

    """Return whether the customer wants exactly one of the three available toppings

    on their hot dog.

    """

    y = bool((ketchup ^ mustard ^ onion)^(ketchup & mustard & onion))

    print("exactly one sauce:",  bool((ketchup ^ mustard ^ onion)^(ketchup & mustard & onion)))

    return y



exactly_one_topping(1, 0, 0)

exactly_one_topping(0, 1, 0)

exactly_one_topping(0, 0, 0)

exactly_one_topping(0, 0, 1)

exactly_one_topping(1, 1, 0)

exactly_one_topping(1, 0, 1)

exactly_one_topping(0, 1, 1)

exactly_one_topping(1, 1, 1)

exactly_one_topping(True, 0, 0)

q6.check()
#q6.hint()

#q6.solution()
def should_hit(player_total, dealer_total, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    if player_total >= 21 or dealer_total >= 21:

        return False

    # Dealer

    if dealer_total >= 17:

        return False

    else:

        return True
q7.simulate_one_game()
q7.simulate(n_games=1000)
def should_hit(player_total, dealer_total, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    return player_total <= 15



q7.simulate(n_games=1000)