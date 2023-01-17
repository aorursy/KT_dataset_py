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
    graph.set_ylabel("Balance")
    
    # getting ticks
    ticks = graph.get_yticks()
    
    # creating new labels (two ways of formatting numbers)
    # labels = ["$" + str(int(tick)) for tick in ticks]
    labels = ["$" + str(tick).rstrip("0").rstrip(".") for tick in ticks]
    
    # setting new labels
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
            
            # my solution - changing internal loop variable name from 'i' to 'item'
            # thus avoiding 'variable shadowing bug' (same variable in both loops)
            
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
# 'helper' function
def score_count(hand):
    
    # creating dictionary of scores
    scores = {'A': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
              '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10}
    
    # counting score
    # initially, each 'Ace' counts for 1 point
    score = 0
    for card in hand:
        score += scores[card]
    
    # recalculating 'Aces'
    # additional 10 points should be applied only if:
    # 1) there are 'Aces' in hand
    for ace in range(hand.count('A')):
        # 2) adding 10pts will not bring total score above 21pts        
        if score <= 11:
            score += 10
    
    # in fact, no matter how many 'Aces' in hand, only one is relevant
    # as calculating two 'Aces' for 11 pts each inevitably breaks 21pts
    
    return score
    
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
    
    # calculating scores using 'helper' function
    score_1 = score_count(hand_1)
    score_2 = score_count(hand_2)   
    
    print()
    print("Hand 1:", hand_1)
    print("No. of Aces:", hand_1.count('A'))
    print("Score:", score_1)

    print("Hand 2:", hand_2)
    print("No. of Aces:", hand_2.count('A'))
    print("Score:", score_2)

    # checking conditions for beat
    beat = False
    if score_1 <= 21 and ((score_1 > score_2) or score_2 > 21):
        beat = True
    return beat

    
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
# my first simple solution
# collecting frequencies of pairs: previous number / last number
# no 'learning phase' =>
# betting for the whole game only on pairs with at least 3 occurences

def my_agent(wheel):
    
    # creating dictionary for relative frequencies
    probs = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 
             6: {}, 7: {}, 8: {}, 9: {}, 10: {},
            }    
    
    last_number = 0
    
    while wheel.num_remaining_spins() > 0:
        
        # remembering last number
        prev_number = last_number
        
        # guessing next bet
        # initially setting not to bet        
        guess = None
        
        # checking relative frequencies for possible followers of previous number
        for bet in probs[prev_number]:
            # betting only on a number that happened at least for 3 times as a follower of previous number
            if probs[prev_number][bet] >= 3:
                guess = bet
                #print("previously", prev_number, "  -  betting on", bet)
                
        # betting        
        last_number = wheel.spin(number_to_bet_on=guess)
        #print(prev_number, last_number)
        
        # adding another pair of previous number / last number to rel. freq. dictionary
        if last_number not in probs[prev_number]:
            probs[prev_number][last_number] = 0
        probs[prev_number][last_number] += 1    
            
#    for i, prob in probs.items():
#        print(i," : ", prob)
            
#help(roulette.RouletteSession)
#help(roulette.evaluate_roulette_strategy)

roulette.evaluate_roulette_strategy(my_agent, num_simulations=20000)