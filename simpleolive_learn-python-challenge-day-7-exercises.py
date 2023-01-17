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
    graph.set_ylim(ymin=0)
    graph.set_ylabel("Balance")
    graph.set_yticklabels(["$0","$50","$100","$150","$200","$250","$300"])
    # Complete steps 2 and 3 here

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
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(i+1, len(racers), racer['name']))
    return winner_item_counts

# Try analyzing the imported full dataset
full_dataset
best_items(full_dataset)
q2.hint()
#q2.solution()
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

    integer_value1 = []
    integer_value2 = []
    
    for i in range(len(hand_1)):
        if hand_1[i] == any(['K','J','Q']):
            integer_value1.append(10)
            
            if hand_1[i] == any(['1','2','3','4','5','6','7','8','9','10']):
                integer_value1.append(int(hand_1[i]))
                
                if hand_1[i] == 'A':
                    integer_value1.append(11)
        
        if hand_1.count('A') == 1 and (sum(integer_value1) > 21 and sum(integer_value1) <= 31):
            hand_1['A'] = hand_1['1']
            
            if hand_1.count('A') == 2 and (sum(integer_value1) > 21 and sum(integer_value1) <= 31):
                hand_1['A','A'] = hand_1['A','1']
                
                if hand_1.count('A') == 2 and (sum(integer_value1) > 31 ):
                    hand_1['A','A'] = hand_1['1','1']
    
    for j in range(len(hand_2)):
        if hand_2[j] == any(['K','J','Q']):
            integer_value2.append(10)
            
            if hand_2[j] == any(['1','2','3','4','5','6','7','8','9','10']):
                integer_value2.append(int(hand_2[j]))
                
                if hand_2[j] == 'A':
                    integer_value2.append(11)
                    
        if hand_2.count('A') == 1 and (sum(integer_value2) > 21 and sum(integer_value2) <= 31):
            hand_2['A'] = hand_2['1']
            
            if hand_2.count('A') == 2 and (sum(integer_value2) > 21 and sum(integer_value2) <= 31):
                hand_2['A','A'] = hand_2['A','1']
                
                if hand_2.count('A') == 2 and (sum(integer_value2) > 31):
                    hand_2['A','A'] = hand_2['1','1']
        
        
    if sum(integer_value1) > sum(integer_value1) and sum(integer_value2) <= 21:
        return True
    else:
        return False
    
    pass


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
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    dic1 = {}
    for i in range(1,len(history)):
        roll,prev = history[i],history[i-1]
        if prev not in dic1:
            dic1[prev] = {}
            
        if roll not in dic1[prev]:
            dic1[prev][roll] = 0
        dic1[prev][roll] += 1
        
        result = {}
    for prev,nexts in dic1.items():
        total = sum(nexts.values())
        sub_result = {nexts_spin:nexts_count/total for nexts_spin,nexts_count in nexts.items()}
        result[prev] = sub_result
        
    #return result  

    pass
        


def my_agent(wheel):
    
    history = []
    last_number = 0
    
    while wheel.num_remaining_spins() > 0:
        history.append(last_number)
    
        dic3 = conditional_roulette_probs(history)
        dic4 = dic3.values()
        prob_values = dic4.values()
        
    for i in range(num_remaining_spins):
        if prob_values[i] > 0.5:
            guess = prob_values[i]
            
        else:
            guess = None
    
        last_number = wheel.spin(number_to_bet_on = guess)
            
    pass

roulette.evaluate_roulette_strategy(my_agent)
