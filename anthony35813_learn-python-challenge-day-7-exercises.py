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
    graph.set_ylim(0,350)
    graph.set_ylabel('Balance') #could just add ($) to this title
    graph.set_yticklabels(['0','$50', '$100', '$150', '$200', '$250', '$300', '$350'])
    

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
            for r in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if r not in winner_item_counts:
                    winner_item_counts[r] = 0
                winner_item_counts[r] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] == None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format( i+1, len(racers), racer['name'] ))
            
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
full_dataset
type(full_dataset)
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
    #first see if hand 1 goes bust?
    #counter for hand 1
    hand1_count = 0
    #need to count aces as only 1 can be worth 11
    ace_count = 0
    for card in hand_1:
        #replace picture cards with values and add up
        if card == 'K' or card == 'J' or card == 'Q':
            hand1_count += 10
        #count aces as 11 first time round
        elif card == 'A' and ace_count < 1:
            hand1_count += 11
            ace_count += 1
        #extra aces can only count as 1
        elif card == 'A' and ace_count >= 1:
            hand1_count += 1
        else:
            # add the face value of the other cards
            hand1_count += int(card)
            
    # if hand1 is over 21 and there is no Ace its all over
    if hand1_count > 21 and 'A' not in hand_1:
        return False
        #break
    #if theres an A we can take 10 off
    elif hand1_count > 21 and 'A' in hand_1:
        hand1_count -= 10
        if hand1_count > 21:
            return False
        #carry on with reduced value of hand 1
    else:
        hand1_count   #left over from printing values to check it was working, take out on final tidy up
    
    #now we need to add up hand 2
    hand2_count = 0
    #need to count aces as only 1 can be worth 11
    ace2_count = 0
    for card2 in hand_2:
        #replace picture cards with values and add up
        if card2 == 'K' or card2 == 'J' or card2 == 'Q':        
            hand2_count += 10
        elif card2 == 'A' and ace2_count < 1:      #count aces as 11 first time round
            hand2_count += 11
            ace2_count += 1
        elif card2 == 'A' and ace2_count >= 1:     #second ace can only be worth 1
            hand2_count += 1
        else:                                      # add the face value of the other cards
            hand2_count += int(card2)
            
    #check to see if hand 2 is bust
    if hand2_count > 21 and 'A' not in hand_2:
        return True
    
    #if > 21 with an A
    elif hand2_count > 21 and 'A' in hand_2:
        hand2_count -= 10
        if hand2_count > 21:
            return True
    else:
        hand2_count
        
    #ok nobodies bust let see who's got the best hand
    if hand1_count > hand2_count:
        return True      # no win if =
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
    res_number = 0
    prev_number = 0
    #  create a dictionary to collect the results
    results = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    while wheel.num_remaining_spins() > 0:
        if len(results[res_number]) > 0:  
            guess = max(results[res_number], key=results[res_number].count)  # take the mode of the list to see which is the most frequent result
        else:
            guess = random.randint(0, 10) # if we dont have any previous results for this number use a random one
        prev_number = res_number
        res_number = wheel.spin(number_to_bet_on=guess)
        results[prev_number].append(res_number)
roulette.evaluate_roulette_strategy(my_agent)