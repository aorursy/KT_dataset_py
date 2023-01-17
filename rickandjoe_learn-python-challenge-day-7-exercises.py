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
    # Complete steps 2 and 3 here
    graph.set_ylim(0)
    
    graph.set_ylabel('Balance')

    graph.set_yticklabels(['$0','$50','$100','$150','$200','$250','$300'])
    
graph = jimmy_slots.get_graph()
prettify_graph(graph)
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
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(i+1, len(racers), str(racer['name']))) 
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
#q2.hint()
#q2.solution()
def total_hand(hand):
    total = 0
    for i in range(len(hand)):
        if hand[i] in ['J','Q','K']:
            total = total + 10
        elif hand[i] in ['A']:
            total = total + 11
        else:
            total = total + int(hand[i])
                
    if total > 21:
        if 'A' in hand:
            ace_counter = 0
            while total > 21 and ace_counter < hand.count('A'):
                total = total - 10
                ace_counter += 1
            
    return total

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
    total_hand1, total_hand2 = total_hand(hand_1), total_hand(hand_2)
    if total_hand1 > 21:
        total_hand1 = -1
    elif total_hand2 > 21:
        total_hand2 = -1
        
    return total_hand1 > total_hand2

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
import numpy as np
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    dictionary = {}
    once = True
    for i in range(len(history)):
        indices =  list(np.where(np.array(history) == history[i])[0])
        if history[i] in dictionary:
            continue
        
        if history[-1] == history[i] and once == True:
            indices.pop()
            once = False
        
        this = [history[index+1] for index in indices]
        if len(this) > 0:
            dictionary[history[i]] = {key: this.count(key)/len(this) for key in this}
    
    return dictionary

def get_lucky_number(cheat_dict):
    """
    Takes a dictionary containing the probabilities of hitting the numbers after from a 
    certain number in a roulette. Returns the lucky number. This assumes that there is
    only one pair of biased numbers.
    """
    highest_prob = 0
    for i in cheat_dict.keys():
        high = sorted(cheat_dict.get(i).values())[-1]
        if high > highest_prob:
            highest_prob = high
            lucky_number = i
            bet_on_this = list(cheat_dict.get(i).keys())[list(cheat_dict.get(i).values()).index(high)]
        
    return lucky_number, bet_on_this
    
    
def my_agent(wheel):
    roulette_history = []
    just_once = True
    while wheel.num_remaining_spins() > 0:
        if wheel.num_remaining_spins() > 40:
            last_number = wheel.spin(number_to_bet_on= None)
            roulette_history.append( last_number ) 
        elif just_once == True:
            lucky_number, bet_on_this  = get_lucky_number(conditional_roulette_probs(roulette_history))
            just_once = False
        else:
            
            if last_number == lucky_number:
                guess = bet_on_this
            else:
                guess = None
            last_number = wheel.spin(number_to_bet_on = guess)
roulette.evaluate_roulette_strategy(my_agent)