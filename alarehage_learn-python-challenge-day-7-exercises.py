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
    graph.set_ylim(0)
    graph.set_ylabel('Balane')

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

#print(help(full_dataset))
#print (full_dataset)
# Try analyzing the imported full dataset
best_items(full_dataset)
q2.hint()
q2.solution()
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
    hand_1_score = 0
    hand_2_score = 0
    hand_1.sort()
    hand_2.sort()
    aces = 0
    
    for card in hand_1:
        if card == 'J' or card == 'Q' or card == 'K':
            hand_1_score += 10
        elif card == 'A':
            aces += 1
        else:
            hand_1_score += int(card)
    while aces > 0:
        if hand_1_score <= 10:
            hand_1_score += 11
        else:
            hand_1_score += 1
        aces -= 1
    aces = 0        
    for card in hand_2:
        if card == 'J' or card == 'Q' or card == 'K':
            hand_2_score += 10
        elif card == 'A':
            aces += 1
        else:
            hand_2_score += int(card)
    while aces > 0:
        if hand_2_score <= 10:
            hand_2_score += 11
        else:
            hand_2_score += 1
        aces -= 1
        
    return hand_1_score <= 21 and (hand_1_score > hand_2_score or hand_2_score > 21)
        

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
def my_agent(wheel):
    last_number = 0
    history = []
    prob = {}
    max_prob = 0
    while wheel.num_remaining_spins() > 0:
        if last_number in prob.keys():
            max_prob = max(prob[last_number].values())
            bet_num = max(prob[last_number], key = prob[last_number].get)
            #bet_num = max([[prob[i], i] for i in prob[last_number]])[1]
            if max_prob > 0.1:
                last_number = wheel.spin(number_to_bet_on = bet_num)
        else:
            bet_num = None
            last_number = wheel.spin(number_to_bet_on = bet_num)
        history.append(last_number)
        prob = conditional_roulette_probs(history)

        
        
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    output = {}
#     ind_prob = {}
    total_count = {}
#     ind_count = {}

    for i in range(1, len(history)):
        roll, prev_roll = history[i], history[i-1]
        if prev_roll not in total_count:
            total_count[prev_roll] = {}
        if roll not in total_count[prev_roll]:
            total_count[prev_roll][roll] = 0
        total_count[prev_roll][roll] += 1
    #print (total_count.values())
    
    prob = {}
    for prev, rolls in total_count.items():
        total = sum(rolls.values())
        sub_prob = {}
        for rolls_no, rolls_count in rolls.items():
                sub_prob[rolls_no] = rolls_count/total
            #    print (sub_prob)
        prob[prev] = sub_prob
    return prob

#print(help(roulette))
roulette.evaluate_roulette_strategy(my_agent)