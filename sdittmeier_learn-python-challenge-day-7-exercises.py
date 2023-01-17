# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex7 import *
print('Setup complete.')
# Import the jimmy_slots submodule
from learntools.python import jimmy_slots
# Call the get_graph() function to get Jimmy's graph
graph = jimmy_slots.get_graph()
graph
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    graph.set_ylim(bottom=0)
    graph.set_ylabel("Balance")
    ticks=graph.get_yticks()
    newlabels = ["${}".format(int(amt)) for amt in ticks]
    graph.set_yticklabels(newlabels)


graph = jimmy_slots.get_graph()
prettify_graph(graph)
#dir(graph)
graph
#q1.solution()
def best_items(racers):
    """Given a list of racer dictionaries, return a dictionary mapping items to the number
    of times those items were picked up by racers who finished in first place.
    """
    winner_item_counts = {}
    for i in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[i]
        # We're only interested in racers who finished in first
        if racer['finish'] == 1:
            for i in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if i not in winner_item_counts:
                    winner_item_counts[i] = 0
                winner_item_counts[i] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
    return winner_item_counts
sample = [
    {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell',], 'finish': 3},
    {'name': 'Bowser', 'items': ['green shell',], 'finish': 1},
    {'name': None, 'items': ['mushroom',], 'finish': 2},
    {'name': 'Toad', 'items': ['green shell', 'mushroom'], 'finish': 1},
]
best_items(sample)
# Import luigi's full dataset of race data
from learntools.python.luigi_analysis import full_dataset

# Fix me!
def best_items(racers):
    winner_item_counts = {}
    for i in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[i]
        # We're only interested in racers who finished in first
        if racer['finish'] == 1:
            for j in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if j not in winner_item_counts:
                    winner_item_counts[j] = 0
                winner_item_counts[j] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
#q2.hint()
#q2.solution()
def max_hand(hand):
    val = 0
    temp = 0
    res = -1
    aces = 0
    
    face_cards = ['J', 'Q', 'K']
    ace = 'A'
    
    for card in hand:
        if(card in face_cards):
            val += 10
        elif(card == ace):
            aces += 1
        else :
            val += int(card)
            
    if(aces == 0):
        res = val # nothing we can do, that's the value
    else: 
        for i in range(aces):
            # ace can be 1 or 11
            temp = val + i*11 + (aces-i)*1
            # if new value is ok (<=21) and larger than what we currently have, store it, if not possible: res will be -1
            if (temp <= 21) and (temp > res):
                res = temp
           # ace can be 1 or 11
            temp = val + (aces-i)*11 + i*1
            # if new value is ok (<=21) and larger than what we currently have, store it, if not possible: res will be -1
            if (temp <= 21) and (temp > res):
                res = temp
    return res
            
        

def blackjack_hand_greater_than(hand_1, hand_2):
    """
    Return True if hand_1 beats hand_2, and False otherwise.
    
    In order for hand_1 to beat hand_2 the following must be true:
    - The total of hand_1 must not exceed 21
    - The total of hand_1 must exceed the total of hand_2 OR hand_2's total must exceed 21
    
    Hands are represented as a list of cards. Each card is represented by a string.
    
    When adding up a hand's total, cards with numbers count for that many points. Face
    cards ('J', 'Q', and 'K') are worth 10 points. 'A' can count for 1 or 11.
    
    When determining a hand's total, you should try to count aces in the way that 
    maximizes the hand's total without going over 21. e.g. the total of ['A', 'A', '9'] is 21,
    the total of ['A', 'A', '9', '3'] is 14.
    
    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    """
    
    val_1 = max_hand(hand_1)
    val_2 = max_hand(hand_2)
    # val_1 is -1: with aces, but larger than 21
    # val_1 is larger than 21: same
    expression = (val_1 != -1) and (not (val_1 > 21)) and ((val_1 > val_2) or val_2 == -1 or val_2 > 21)
    print(hand_1, hand_2, val_1, val_2, expression)
    return expression    

q3.check()
#q3.hint()
#q3.solution()
from learntools.python import roulette
import random

def random_and_superstitious(wheel):
    """Interact with the given wheel over 100 spins with the following strategy:
    - if the wheel lands on 4, don't bet on the next spin
    - otherwise, bet on a random number on the wheel (from 0 to 10)
    """
    last_number = 0
    while wheel.num_remaining_spins() > 0:
        if last_number == 4:
            # Unlucky! Don't bet anything.
            guess = None
        else:
            guess = random.randint(0, 10)
        last_number = wheel.spin(number_to_bet_on=guess)

roulette.evaluate_roulette_strategy(random_and_superstitious)
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    last_spin = -1
    dictionary = {}
    dictionary_inside = {}
    for spin in history:
        if last_spin != -1:
            if(last_spin in dictionary):
                dictionary_inside = dictionary[last_spin]
                if(spin in dictionary_inside):
                    dictionary_inside[spin] += 1.0
                else:
                    dictionary_inside[spin]= 1.0
            else:
                dictionary_inside = {spin: 1.0}
            dictionary[last_spin] = dictionary_inside
        last_spin = spin
        
    for key in dictionary:
        sum = 0
        for key_inside in dictionary[key]:
            sum += dictionary[key][key_inside]
        for key_inside in dictionary[key]:
            dictionary[key][key_inside] = dictionary[key][key_inside]/sum
            
    return dictionary

def find_max(dictionary,key):
    if key in dictionary :
        return max(dictionary[key],key=dictionary[key].get)
    else: 
        return -1
        
    dict_inside = dictionary[key]

def my_agent(wheel):
    last_number = 0
    history = []
    guess = None
    probs = conditional_roulette_probs(history)
    #while wheel.num_remaining_spins() > 50:
    #    last_number = wheel.spin(number_to_bet_on=guess)
    #    history.append(last_number)
    #probs = conditional_roulette_probs(history)
    #while wheel.num_remaining_spins() > 0:
    #    guess = find_max(probs,last_number)
    #    if guess == -1:
    #        guess = None
    #    last_number = wheel.spin(number_to_bet_on=guess)
    while wheel.num_remaining_spins() > 0:
        guess = find_max(probs,last_number)
        if guess == -1:
            guess = None
        last_number = wheel.spin(number_to_bet_on=guess)
        history.append(last_number)
        probs = conditional_roulette_probs(history)

#help(roulette.RouletteSession)
#help(roulette.evaluate_roulette_strategy)
roulette.evaluate_roulette_strategy(my_agent)