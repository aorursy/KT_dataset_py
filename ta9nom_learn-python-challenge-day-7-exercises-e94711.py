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
                    winner_item_counts[i] = 0
                winner_item_counts[i] += 1

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
    def BlackjackHand(hands):
        res = 0
        cntA = 0
        for hand in hands:
            if hand in ['J','Q','K']:
                res = res + 10
            if hand in ['1','2','3','4','5','6','7','8','9','10']:
                res = res + int(hand)
            if hand == 'A':
                cntA = cntA + 1
        if cntA == 0:
            pass
        if cntA == 1:
            if res <= 10:
                res = res + 11
            else:
                res = res + 1
        if cntA == 2:
            if res <= 9:
                res = res + 12
            else:
                res = res + 2
                
    hand1 = BlackjackHand(hand_1)
    hand2 = BlackjackHand(hand_2)
    
    if hand1 <= 21:
        if (hand1 > hand2) or (hand2 > 21):
            return True
        else:
            return False
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
def my_agent(wheel):
    his_search = []
    n_his_search = int(wheel.num_remaining_spins()*70/100)

    for i in range(n_his_search):
        his_search.append(wheel.spin())
    
    his ={}
    for i,v in enumerate(his_search):
        if v not in his:
            his[v] = {}
        if i < (len(his_search) - 1):
            his[v][his_search[i+1]] = 0
    for k in his:
        l = len(his[k])
        for x in his[k]:
            his[k][x] = 1 / l
    
    highrate = 0
    next_num = 0
    cur_num = 0
    
    for k, v in his.items():
        for t, d in v.items():
            if d > highrate:
                highrate = d
                next_num = t
                cur_num = k
#    print(highrate)
#    print(next_num)
#    print(cur_num)
    
    last_num = 0
    guess = None
    while wheel.num_remaining_spins() > 0:
        if last_num == cur_num:
            guess = next_num
        else:
            guess = None
        last_num = wheel.spin(number_to_bet_on=guess)

#a={1: {3: 0.5, 5: 0.5}, 
#       3: {1: 1.0},
#       5: {1: 1.0}
#  }

#wheel = roulette.RouletteSession(11,100,0.5)
#wheel.spin()

#help(roulette)
#help(roulette.RouletteSession)
#help(roulette.evaluate_roulette_strategy)
roulette.evaluate_roulette_strategy(my_agent, num_spins_per_simulation=700)