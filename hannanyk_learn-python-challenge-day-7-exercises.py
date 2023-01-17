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
# help(graph.set_ylim)
# dir(graph)
# help(graph.set_yticklabels)
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    graph.set_ylim(bottom = 0)
    graph.set_ylabel("Balance")
    orig_yticks = graph.get_yticks()
    new_yticklabels = ["${}".format(number) for number in orig_yticks]
    graph.set_yticklabels(new_yticklabels)

graph = jimmy_slots.get_graph()
prettify_graph(graph)
graph
# q1.solution()
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
q2.solution()
hand = ['A', 'A', '9', '3']
hand.sort()
def count_blackjack(hand):
    sorted_hand = sorted(hand)
    aces = 0
    if "A" in sorted_hand:
        aces = sorted_hand.count("A")
        for i in range(aces):
            sorted_hand.remove("A")
    hand_sum = 0
    for card in sorted_hand:
        if (card == "J") or (card == "Q") or (card == "K"):
            hand_sum += 10
        else:
            hand_sum += int(card)
    for i in range(aces + 1):
        temp = hand_sum + (aces - i) * 11 + i * 1
        if temp < 21:
            return temp
    return hand_sum + aces 
        
    
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
    count_1 = count_blackjack(hand_1)
    count_2 = count_blackjack(hand_2)
    if (count_1 <= 21) and ((count_1 > count_2) or (count_2 > 21)):
        return True
    else:
        return False

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
help(roulette.RouletteSession)
dict_1 = {1: {3: 0.5, 5: 0.5}, 
          3: {1: 1.0},
          5: {1: 1.0},}
len(dict_1[1])
def unique(list_object):
    unique_list = []
    for element in list_object:
        if not element in unique_list:
            unique_list.append(element)
    return unique_list
    
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    keys = unique(history)
    roulette_dict = {}
    for key in keys:
        keys_n2 = []
        for appearance in [ind for ind in range(len(history[:-1])) if history[ind] == key]:
            keys_n2.append(history[appearance + 1])
        mini_dict = {}
        if not len(unique(keys_n2)) == len(keys_n2):
            mini_dict = {key_n2: keys_n2.count(key_n2) / len(keys_n2) for key_n2 in keys_n2}
        else:
            mini_dict = {key_n2: 1/(len(keys_n2)) for key_n2 in keys_n2}
        roulette_dict[key] = mini_dict
    return roulette_dict

def get_high_prob_pair(conditional_prob):
    best_overall = 0
    best_pair = None
    for key in conditional_prob.keys():
        best_prob = 0
        following_number = 0
        for key_n2, value in conditional_prob[key].items():
            if value > best_prob:
                best_prob = value
                following_number = key_n2
        if best_prob > best_overall:
            best_overall = best_prob
            best_pair = [key, key_n2]
    return best_pair, best_overall

def my_agent(wheel):
    history = [wheel.spin() for i in range(25)]
    conditional_prob = conditional_roulette_probs(history)
    best_pair, best_overall = get_high_prob_pair(conditional_prob)
    last_number = history[-1]
    while wheel.num_remaining_spins() > 0:
        if last_number == best_pair[0]:
            guess = best_pair[1]
        else:
            guess = None
        last_number = wheel.spin(number_to_bet_on=guess)
        

roulette.evaluate_roulette_strategy(my_agent)