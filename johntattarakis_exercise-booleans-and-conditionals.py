from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
# Your code goes here. Define a function called 'sign'
def sign2(n):
    if n == 0:
        return 0
    elif n > 0:
        return 1
    else:
        return -1

def sign3(n):
    return 0 if n == 0 else n/abs(n)

def sign(n):
    return 0 if n == 0 else (1 if n > 0 else -1)
    
# Check your answer
q1.check()
q1.solution()
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
    print("Splitting", total_candies, "cand" + ("y" if total_candies == 1 else "ies"))
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
rain_level = 1
have_hood = False
is_workday = False

# Check what the function returns given the current values of the variables above
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

# Check your answer
q3.check()
#q3.hint()
q3.solution()
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

def concise_is_negative(number):
    return number < 0

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
    return ketchup and mustard and onion

# Check your answer
q5.a.check()
#q5.a.hint()
#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):
    """Return whether the customer wants a plain hot dog with no toppings.
    """
    return not (ketchup or mustard or onion)
    
# Check your answer
q5.b.check()
#q5.b.hint()
#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    return ketchup ^ mustard

# Check your answer
q5.c.check()
#q5.c.hint()
q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    return (ketchup + mustard + onion) == 1

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
# Use a persistent dictionary to record intermediate results and avoid multiple equivalent calculations
theDict = {}

def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay.
    When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
    doesn't bring the total above 21, otherwise we count them as low (with value 1). 
    For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
    and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
    """    
    
    # Probability of player winning if they stick
    p_win_if_stick = p_win(dealer_total, player_total, 0, player_high_aces, 1)
    
    # Probability of player winning if they hit
    p_win_if_hit = p_win(dealer_total, player_total, 0, player_high_aces)

    # Return true if hitting is advantageous   
    return p_win_if_hit > p_win_if_stick

# Returns the probability of the player winning
def p_win(dealer_total, player_total, dealer_high_ace = 0, player_high_ace = 0, stick = 0):
    """
    This helper function calculates the probability of the player winning.
    If stick == 1 then only the first half of the function is run, which keeps the player's
    score fixed and varies the dealer's score.
    """
    
    global theDict

    # The dictionary key should be an amalgamation of the arguments to the function
    theKey = ','.join(map(str, locals().values()))

    if theKey in theDict.keys():
        return theDict[theKey]

    ## First explore basic cases

    if player_total > 21:
        # Did player bust?
        if player_high_ace == 1:
            # Player has a high ace, so we can continue
            player_total = player_total - 10
            player_high_ace = 0
        else:
            # Player bust
            return 0

    if dealer_total > 21:
        # Did dealer bust?
        if dealer_high_ace == 1:
            # Dealer has a high ace, so we can continue
            dealer_total = dealer_total - 10
            dealer_high_ace = 0
        else:
            # Dealer bust
            return 1

    # Dealer sticks?
    if dealer_total >= 17:
        return int(player_total > dealer_total)

    # This is just in case the data is given to us poorly
    if dealer_total == 1:
        dealer_total = 11
        dealer_high_ace = 1
        
    ## Next, calculate the probability of player winning if they stick
    
    # Determine if a hypothetical ace is worth 1 or 11
    if dealer_total > 10:
        win_if_stick = p_win(dealer_total + 1, player_total, dealer_high_ace, player_high_ace, 1)
    else:
        win_if_stick = p_win(dealer_total + 11, player_total, 1, player_high_ace, 1)

    win_if_stick = (win_if_stick + 
                    p_win(dealer_total + 2, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 3, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 4, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 5, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 6, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 7, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 8, player_total, dealer_high_ace, player_high_ace, 1) +
                    p_win(dealer_total + 9, player_total, dealer_high_ace, player_high_ace, 1) +
                    # The incidence of cards worth 10 is four times greater than the previous cards
                    4 * p_win(dealer_total + 10, player_total, dealer_high_ace, player_high_ace, 1)) / 13
    
    # Stop now and return if the function is used purely for the above subcalculations
    if stick == 1:
        theDict[theKey] = win_if_stick
        return win_if_stick
    
    ## Finally, calculate the probability of the player winning if they hit

    # Again, determine if a hypothetical ace is worth 1 or 11
    if player_total > 10:
        win_if_hit = p_win(dealer_total, player_total + 1, 0, player_high_ace)
    else:
        win_if_hit = p_win(dealer_total, player_total + 11, 0, 1)
    
    win_if_hit = (win_if_hit + 
                    p_win(dealer_total, player_total + 2, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 3, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 4, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 5, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 6, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 7, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 8, 0, player_high_ace) +
                    p_win(dealer_total, player_total + 9, 0, player_high_ace) +
                    # The incidence of cards worth 10 is four times greater than the previous cards
                    4 * p_win(dealer_total, player_total + 10, 0, player_high_ace)) / 13
    
    # The probability of winning is thus the maximum of the probabilities of either hitting or sticking
    theDict[theKey] = max(win_if_hit, win_if_stick)
    return max(win_if_hit, win_if_stick)

#print(should_hit(11, 12, 0, 0))
q7.simulate(n_games=100000)