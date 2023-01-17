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
graph.set_title("Results of 500 slot machine pulls")
graph.set_ylim(0)
graph.set_ylabel("Balance")
graph.set_yticklabels
prettify_graph(graph)
graph.xlabel(r"$\pounds$")
help(graph.set_)
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
                i+1, 
                len(racers), 
                racer['name'])
                 )
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
#q2.hint()
#q2.solution()
def hand_value (hand_cards):
    value = 0
    asses = 0
    for card in hand_cards:
        value_card = 0
        if card in ["2","3","4","5","6","7","8","9","10"]: value_card = int(card)
        if card in ["J","Q","K"]: value_card = 10
        if card == "A": asses += 1
        value = value + value_card
    if value + asses > 21: return 500
    elif asses == 0: return value
    else:
        if asses == 1: asses_variants = (1+value,11+value)
        if asses == 2: asses_variants = (2+value,12+value,22+value)
        if asses == 3: asses_variants = (3+value,13+value,23+value)
        if asses == 4: asses_variants = (4+value,14+value,24+value,34+value)
    variant_out = 500
    for variant in asses_variants:
        if variant - 22 < 0: variant_out = variant
        else: return variant_out
    return variant_out
hand_value (['A', '10', 'K'])

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
#     pass
    if hand_value (hand_1) == 500: return False
    if hand_value (hand_2) == 500: return True 
    return hand_value (hand_1) > hand_value (hand_2)
            
# blackjack_hand_greater_than(['K'], ['10'])

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
help(roulette.RouletteSession)
from learntools.python.roulette import *
roulete_session = RouletteSession(10,100,0.5)
roulete_session.spin(9)
roulete_session.spin(9)
roulete_session.num_remaining_spins()
roulete_session.evaluate_roulette_strategy
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
#     pass
    res_list=[]
    res_map={}
    number_list = list(set(history))
#     print (number_list)
#     for i in number_list:
#         for n in range(len(history)):
#             if history[n] == i:
#                 if n+1 < len(history):
# #                     res_list.append([history[n],history[n+1]])
#                     if res_map[history[n],history[n+1]] is NONE:
#                         res_map[history[n],history[n+1]] = 1
#                     else:
#                         res_map[history[n],history[n+1]] = res_map[history[n],history[n+1]] + 1

    
    for n in range(len(history)):
        if n+1 < len(history):
            res_list.append([history[n],history[n+1]])
#     for l in res_list
#         if l in res_list
#             res_list_count = 
#     return res_list
    dict_out ={}
    for n in number_list:
        dict_in = {}
        for m in res_list:
#             print("n",n)
#             print("m",m)
#             print ("m[0]",m[0])
#             print(m[0]==n)
            if m[0]==n:
#                 print("in")
#                 print(m[1])
#                 print(dict_in[3])
#                 print(m[1] in dict_in)
                if m[1] not in dict_in: 
                    dict_in[m[1]] = 1
#                     print("a",dict_in)
                else:
                    dict_in[m[1]] = dict_in[m[1]] + 1
        sum_dict_in = 0;
        for k,v in dict_in.items():
            sum_dict_in = sum_dict_in + v
#         print("sum_dict", sum_dict_in)
        for k,v in dict_in.items():
            dict_in[k] = v/sum_dict_in
        dict_out[n] = dict_in
#     print (dict_out)
    return(dict_out)
conditional_roulette_probs([1,9])
def my_agent(wheel):
#     pass
    bets_results=[]
    number_to_bet_on = 1
    for i in range (100):
        result = wheel.spin(number_to_bet_on=number_to_bet_on)
        bets_results.append(result)
        probability_dict = conditional_roulette_probs(bets_results)
        if probability_dict[result] == {}: number_to_bet_on = 1
        else: number_to_bet_on = max(probability_dict[result], key=probability_dict[result].get)
        #     print (bets_results)
        #     print(conditional_roulette_probs(bets_results))
  
roulette.evaluate_roulette_strategy(my_agent)