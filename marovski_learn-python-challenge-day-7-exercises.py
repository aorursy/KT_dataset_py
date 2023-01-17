# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex7 import *
print('Setup complete.')
# Import the jimmy_slots submodule
from learntools.python import jimmy_slots
# Call the get_graph() function to get Jimmy's graphdir(graph)
graph = jimmy_slots.get_graph()
graph
import matplotlib.ticker as mtick
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
   
    # Complete steps 2 and 3 here
    graph.set_ylabel("Balance")
    graph.set_ylim(0)
    graph.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    
    
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
q2.hint()
q2.solution()

def blackjack_hand_greater_than(hand_1, hand_2):
    value_1=0
    
    value_2=0
    """
    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    """
    
  
    
    for i in hand_1:
        if i=='K' or i=='J' or i=='Q':
            value_1+=10
        elif i=='A' and value_1<=21 and value_1 >= 11 :
            value_1+=1
        elif i=='A' and value_1<=21 and value_1 <= 10:
            value_1+=11
            
        else:
            value_1+=int(i)
           
    for i in hand_2:
        
        if i=='K' or i=='J' or i=='Q':
            value_2+=10
        elif i=='A' and value_2<=21 and value_2>=11 :
            value_2+=1
        elif i=='A' and value_2<=21 and value_2<=10:
            value_2+=11
            
        else:
            value_2+=int(i)
    
    
    return value_1 <= 21 and (value_1 > value_2 or value_2 > 21)


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


# function from day_6 
def conditional_roulette_probs(history):
    summ = {}
    for i in range(len(history)-1):
        aft, bef = history[i+1], history[i]
        if bef not in summ:
            summ[bef] = {}
        if aft not in summ[bef]:
            summ[bef][aft] = 0
        summ[bef][aft] += 1

    prob = {}
    for bef, aft in summ.items():
        sum_all = sum(aft.values())
        local_prob = {k: v/sum_all
                for k, v in aft.items()}
        prob[bef] = local_prob
    return prob

def my_agent(wheel):
    last_number = 0
    hist = []
    prob = {}
    spins = wheel.num_remaining_spins()
    while wheel.num_remaining_spins() > 0:
        if last_number in prob.keys():
            guess = max(prob[last_number], key = prob[last_number].get)
        else:
            guess = None
        last_number = wheel.spin(number_to_bet_on=guess)
        hist.append(last_number)
        if wheel.num_remaining_spins() % 10 == 0:
            prob = conditional_roulette_probs(hist)

# help(roulette.RouletteSession)
# help(roulette.evaluate_roulette_strategy)
roulette.evaluate_roulette_strategy(my_agent, biased_transition_prob=1)