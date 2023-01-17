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
    graph.set_ylim(bottom=0)
    graph.set_ylabel("Balance")
    y_labels = graph.get_yticks()
    #print('y_labels: ', y_labels)
    #graph.set_yticklabels(['$%d' % int(y) for y in y_labels]) #was my solution
    graph.set_yticklabels(['${}'.format (int(y)) for y in y_labels])
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
    def hand_val(hand):
        """
        Return blackjack value of hand
        Examples:
        >>> hand_val(['K', 'K', '2'])
        22
        >>> hand_val(['A', 'A', '9'])
        21
        """
        current_hand_val = 0
        ace_bonus_avail = False
        for i in range(len(hand)):
            if hand[i] in ['J', 'Q', 'K']:
                current_hand_val += 10
            elif hand[i] == 'A':
                current_hand_val += 1
                ace_bonus_avail = True
            else:
                current_hand_val += int(hand[i])
        if current_hand_val <= 11 and ace_bonus_avail:
            current_hand_val += 10
        return(current_hand_val)

    hand1_val = hand_val(hand_1)
    #print('hand1_val: ', hand1_val)
    hand2_val = hand_val(hand_2)
    #print('hand2_val: ', hand2_val)
    if (hand1_val <= 21) and ((hand1_val > hand2_val) or (hand2_val > 21)):
        return(True)
    else:
        return(False)

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
#help(roulette.RouletteSession)
help(roulette.evaluate_roulette_strategy)
def my_agent(wheel):
    """Interact with the given wheel over 100 spins with the following strategy:
    - If the wheel lands on 4, don't bet on the next spin
    - otherwise, bet on a random number on the wheel (from 0 to 10)
    """
    def conditional_roulette_probs(history):
        """
    
        Example: 
        conditional_roulette_probs([1, 3, 1, 5, 1])
        > {1: {3: 0.5, 5: 0.5}, 
           3: {1: 1.0},
           5: {1: 1.0}
          }
        """
        d_current_spin = {}
        d_current_spin_count = {}
        d_result = {}
        next_spin = history.pop(0)
        while (len(history) > 0):
            current_spin = next_spin
            next_spin = history.pop(0)
            if current_spin not in d_current_spin:
                d_current_spin[current_spin] = {}
                d_current_spin[current_spin][next_spin] = 1
            else:
                if next_spin not in d_current_spin[current_spin]:
                    d_current_spin[current_spin][next_spin] = 1
                else:
                    d_current_spin[current_spin][next_spin] += 1
            if current_spin not in d_current_spin_count:
                d_current_spin_count[current_spin] = 1
            else:
                d_current_spin_count[current_spin] +=1
        for k in d_current_spin.keys():
            d_result[k] = {}
            for (m, n) in d_current_spin[k].items():
                d_result[k][m] = (n / d_current_spin_count[k])
        return (d_current_spin_count, d_result)

    last_number = 0
    spin_list = []
    spin_probs = {}
    guess = None
    bet_thru_nth_probability = 3
    nth_highest_spin_probability = 7
    num_followed_by_spins_per_initial_spin = 1
    while wheel.num_remaining_spins() > 0:
        last_number = wheel.spin(number_to_bet_on=guess)
        spin_list.append(last_number)
        if len(spin_list) >= 20:
            temp_spin_list = spin_list.copy()
            current_spin_count, probs = conditional_roulette_probs(temp_spin_list)
            cnt_initial_spin_has_adequate_num_followed_by_spin = 0
            if len(probs) >= 11:
                for i in range(len(probs)):
                    if (len(probs[i]) >= num_followed_by_spins_per_initial_spin):
                        cnt_initial_spin_has_adequate_num_followed_by_spin += 1
            if cnt_initial_spin_has_adequate_num_followed_by_spin >= 10:                   
                ordered_spin_count = sorted(current_spin_count.items(), key=lambda x: (x[1],x[0]), reverse=True)
                count_highest_spin_probabilities = 0
                highest_spin_probabilites = []
                last_highest_spin_count = 0
                for i in range(len(ordered_spin_count)):
                    if ordered_spin_count[i][1] != last_highest_spin_count:
                        last_highest_spin_count = ordered_spin_count[i][1]
                        count_highest_spin_probabilities += 1
                    if count_highest_spin_probabilities > nth_highest_spin_probability:
                        break
                    highest_spin_probabilites.append(ordered_spin_count[i][0])
                for k in probs:
                    for nk, nv in probs[k].items():
                        spin_probs[(k, nk)] = nv
                pair_prob = sorted(spin_probs.items(), key=lambda x: (x[1],x[0]), reverse=True)
                count_distinct_probabilities = 0
                last_probability = 9
                guess = None
                for i in range(len(pair_prob)):
                    if (pair_prob[i][1] < last_probability):
                        last_probability = pair_prob[i][1]
                        count_distinct_probabilities += 1
                    if (count_distinct_probabilities > bet_thru_nth_probability):
                        break
                    if ((pair_prob[i][0][0] == last_number)
                    and (pair_prob[i][0][1] in highest_spin_probabilites)):
                        guess = pair_prob[i][0][1]
                        break

roulette.evaluate_roulette_strategy(my_agent)