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
dir(graph)
help(graph.get_yticklabels)
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    graph.set_ylim(bottom=0)
    graph.set_ylabel("Balance")
    labels = []
    for tick in graph.get_yticks():
        labels.append("${}".format(int(tick)))
    graph.set_yticklabels(labels)
graph = jimmy_slots.get_graph()
prettify_graph(graph)
graph
q1.solution()
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
q2.hint()
print(full_dataset)
q2.solution()
def value_of_hand_replacing_by(hand, n):
    for i in range(len(hand)):
        if hand[i] == 'A':
            hand[i] = n
            break
    return value_of_hand(hand)

def value_of_hand(hand):
    if 'A' in hand:
        value1 = value_of_hand_replacing_by(list(hand), 1)
        value2 = value_of_hand_replacing_by(list(hand), 11)
        if value1 > value2:
            return value1
        else:
            return value2
    else:    
        value = 0
        for card in hand:
            if type(card) == int or str.isdigit(card):
                value += int(card)
            else:
                value += 10
    if value > 21:
        return 0
    else:
        return value
        
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
    value1 = value_of_hand(hand_1)
    value2 = value_of_hand(hand_2)
    #print("{}({}) vs {}({}) : {}".format(hand_1, value1, hand_2, value2, value1 > value2))
    return value1 > value2

#value_of_hand(['A', 9, 'A'])
q3.check()
#q3.hint()
q3.solution()
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
help(roulette.evaluate_roulette_strategy)
def conditional_roulette_probs(history):
    """
    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    result = {}
    if len(history) > 0:
        prev = history[0]
        for r in history[1:]:
            if not prev in result:
                result[prev] = {}
            if not r in result[prev]:
                result[prev][r] = 1
            else:
                result[prev][r] += 1
            prev = r
        for i, liste in result.items():
            total = sum(result[i].values())
            for n, nb in liste.items():
                result[i][n] = nb / total
    return result

def my_agent(wheel):
    last_number = 0
    history = []
    while wheel.num_remaining_spins() > 0:
        roulette_probs = conditional_roulette_probs(history)
        if last_number in roulette_probs.keys():
            #print('last_number={} history={}'.format(last_number, history))
            #print(roulette_probs)
            max_value = 0
            max_key = 0
            for key, value in roulette_probs[last_number].items():
                if value > max_value:
                    max_value = value
                    max_key = key
            last_number = wheel.spin(number_to_bet_on=max_key)
        else:
            last_number = wheel.spin() # no history, no bet
        history.append(last_number)

roulette.evaluate_roulette_strategy(my_agent)