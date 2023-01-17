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
from matplotlib.ticker import StrMethodFormatter
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    # Complete steps 2 and 3 here
    graph.set_ylim(0,400)
    graph.set_ylabel("Balance")

    graph.yaxis.set_major_formatter(StrMethodFormatter("${x:2.2f}"))
    graph.xaxis.set_major_formatter(StrMethodFormatter("${x:2.2f}"))
    
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
    for ri in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[ri]
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
                ri+1, len(racers), racer['name'])
                 )
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
#q2.hint()
#q2.solution()
from itertools import combinations_with_replacement
def cardeval(c):
    """
    resolve cards that are not fuzzy
    """
    try:
        return(int(c))
    except ValueError:
        if c in 'JQK':
            return(10)
    return(c)

def reduceCards(cl):
    """
    reduce cards to total of know cards plus number of fuzzy cards
    """
    result = []
    facesum = sum([cardeval(c) for c in cl if c != 'A'])
    acecount = len([c for c in cl if c == 'A'])
    return(facesum, acecount)

def handvalue(hand):
    """
    for eac combination of fuzzy cards below 21, find the maximum of known cards plus fuzzy cards that is less than 21
    """
    (tot, ac) = reduceCards(hand)
    val = [sum(list(t)) for t in combinations_with_replacement([1,11],ac) if sum(list(t)) < 21]
    pval = [tot+av for av in val if tot+av <= 21]
    if len(pval) <= 0:
        return 0
    return(max(pval))

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
    return(handvalue(hand_1) > handvalue(hand_2))

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
import pprint
from collections import defaultdict 
def evalmemory(m,spincount):
    rslt = {}
    for fn,rslt in enumerate(m):
        for sn, cnt in enumerate(rslt):
            if(cnt > 0):
                print("{}-{}:{}".format(fn,sn,cnt/spincount))
            
def updatememory(m,fspin, sspin):
    if fspin < 0:
        return
    m[fspin][sspin] += 1

def findbestpair(m,spincount):
    ip = []
    for fn,rslt in enumerate(m):
        for sn, cnt in enumerate(rslt):
            if(cnt > 0 and spincount > 0):
                pairprob = cnt/spincount
                ip.append((fn,sn,pairprob))
    ##sort the ip by pairprob
    rslt = sorted(ip,key=lambda x: x[2],reverse=True)
    return(rslt)

    
def my_agent(wheel):
    ##initilize memory
    #pprint.pprint(vars(wheel))
    
    memory = [[0 for _ in range(wheel._wheel_size)] 
              for _ in range(wheel._wheel_size)]
    
    last_result = -1
    observe = 10
    si = 0
    while wheel.num_remaining_spins() > 0:
        memorder = findbestpair(memory,si)
        beton = None
        if si > observe and last_result == memorder[0][0] and memorder[0][2] > 0.01:
            beton = memorder[0][1]
            #print("last spin {} Betting on {} because {}".format(last_result, beton, memorder[0]))
        current_result = wheel.spin(number_to_bet_on=beton)
        updatememory(memory, last_result, current_result)
        
        last_result = current_result
        si += 1
    #evalmemory(memory,si)
    
    #pprint.pprint(memorder[0:4])

    
roulette.evaluate_roulette_strategy(my_agent)
##mrs = roulette.RouletteSession(11,1000,0.5)
##my_agent(mrs)
#help(roulette.RouletteSession)