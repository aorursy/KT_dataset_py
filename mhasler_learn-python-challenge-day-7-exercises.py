# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex7 import *
print('Setup complete.') ##################### Yes, go ahead: this is the right Kernel !
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
    graph.set_ylabel("Balance")
    # Prefix '$' to the y-tick labels. NB: the tick labels are floats!
    graph.set_yticklabels(['$'+str(y) for y in graph.get_yticks()])

graph = jimmy_slots.get_graph()
prettify_graph(graph)
graph
q1.solution()
# buggy code deleted, see below for correct version
sample = [
    {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell',], 'finish': 3},
    {'name': 'Bowser', 'items': ['green shell',], 'finish': 1},
    {'name': None, 'items': ['mushroom',], 'finish': 2},
    {'name': 'Toad', 'items': ['green shell', 'mushroom'], 'finish': 1},
]
best_items(sample)
# Import luigi's full dataset of race data
from learntools.python.luigi_analysis import full_dataset

# Fix me! -- Done (MFH, 11.1.2019)
def best_items(racers):
    winner_item_counts = {}
    for i in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[i]
        # We're only interested in racers who finished in first
        if racer['finish'] == 1:
            for it in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if it not in winner_item_counts:
                    winner_item_counts[it] = 0
                winner_item_counts[it] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})"
                  .format(i+1, len(racers), racer['name'])
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
    def score(c):
        if c.isdigit(): return int(c); 
        return 10+(c=='A')
    def val(h):
        s = sum(score(c) for c in h)
        if s > 21:
            a = h.count('A')
            while a > 0 and s > 21:
                s -= 10
                a -= 1
        return s
    v1 = val(hand_1)
    if v1 > 21: return False
    v2 = val(hand_2)
    return v1 > v2 or v2 > 21 

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
def my_agent(wheel): # v2.0 - 11.1.2018 by MFH 
    h = {}  # history: dictionary of how many times any roll B
            # happened after a given roll A.
    prev = None # last (preceding) roll
    def my_guess():
        if prev in h:
            if len(h[prev]) > 1:
                # At least 2 distinct outcomes are recorded.
                # Sort them according to decreasing frequency.
                # (It would be more efficient to use a skip_dict
                # which always remains sorted.)
                best = sorted( h[prev], key=h[prev].__getitem__, reverse=True) 
                # Is the difference between most frequent one
                # and second most frequent one large enough?
                if h[prev][best[0]] - h[prev][best[1]] > 0:
                    return best[0] # Then we guess this.
                    # If we make this threshold "delta" (here: 0) larger, the
                    # success rate is higher but gain is lower, since fewer
                    # bets are made. (delta >= 3  =>  success rate >= 50% !) 
            else: # only one result recorded. 
                # Guess this if it happened "often enough" - but here again,
                # smaller threshold gives lower success rate but higher gain.    
                if list(h[prev].values())[0] > 0: 
                    return list(h[prev])[0]       
        return None
    # main
    while wheel.num_remaining_spins() > 0:
        r = wheel.spin(number_to_bet_on = my_guess())   
        # update the history
        if prev:
            if not prev in h: h[prev]={}
            if not r in h[prev]: h[prev][r] = 1
            else: h[prev][r] += 1
        prev = r

roulette.evaluate_roulette_strategy(my_agent)