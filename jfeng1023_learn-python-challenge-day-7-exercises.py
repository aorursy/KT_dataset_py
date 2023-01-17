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
type(graph)
help(graph)
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    # Complete steps 2 and 3 here
    graph.set_ylim(0, 350)
    graph.set_ylabel('Balance')
    # add the $ sign to y-axis labels
    yticks = graph.get_yticks()
    new_labels = ['${}'.format(int(n)) for n in yticks]
    graph.set_yticklabels(new_labels)

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
            for item in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if item not in winner_item_counts:
                    winner_item_counts[item] = 0
                winner_item_counts[item] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
                 
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
print(full_dataset)
q2.hint()
q2.solution()
def calculate_hand(hand):
    """
    Return the total of a hand
    """
    n_ace = hand.count('A')
    total = 0
    for card in hand:
        if card == 'A':
            total += 11
        elif card in ['J', 'Q', 'K']:
            total += 10
        else:
            total += int(card)
    # if the total is over 21 and there's at least one A, count A as 1 as needed
    if total > 21 and n_ace > 0:
        ace_remaining = n_ace
        while total > 21 and ace_remaining > 0:
            total -= 10
            ace_remaining -= 1
    
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
    hand_1_total = calculate_hand(hand_1)
    hand_2_total = calculate_hand(hand_2)
    
    return hand_1_total <= 21 and (hand_1_total > hand_2_total or hand_2_total > 21)

q3.check()
q3.hint()
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
def my_agent(wheel):
    import operator
    
    dict_freq = {}
    last_number = None
    while wheel.num_remaining_spins() > 0:
        # don't bet in the first 30 spins, or if the last number hasn't appeared before
        if wheel.num_remaining_spins() > 70 or last_number not in dict_freq:
            curr_number = wheel.spin()
        else:
            # make a bet based on conditional probability
            most_common_num, highest_freq = max(dict_freq[last_number].items(), key=operator.itemgetter(1))
            if highest_freq/sum(dict_freq[last_number].values()) >= 0.35:
                guess = most_common_num
            else:
                guess = None
            curr_number = wheel.spin(number_to_bet_on=guess)
        # record result
        if last_number != None:
            if last_number not in dict_freq:
                dict_freq[last_number] = {curr_number: 1}
            elif curr_number in dict_freq[last_number]:
                dict_freq[last_number][curr_number] += 1
            else:
                dict_freq[last_number][curr_number] = 1
        last_number = curr_number
    
        
roulette.evaluate_roulette_strategy(my_agent)