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

    graph.set_ylim([0, 350])

    graph.set_ylabel("Balance")

    labels = graph.get_yticks()

    new_labels = ["${}".format(int(value)) for value in labels]

    print(new_labels)

    graph.set_yticklabels(new_labels)

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



# Fix me

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

            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(

                i+1, len(racers), racer['name'])

                 )

    return winner_item_counts



# Try analyzing the imported full dataset

best_items(full_dataset)
#q2.hint()
#q2.solution()
def blackjack_hand(hand):

    royals = ['J','Q','K']

    total_h = 0

    aces_counter_h = 0

    

    for card in hand:

        if card.isnumeric():

            total_h += int(card)

        else:

            if card in royals:

                total_h += 10

            else:

                aces_counter_h += 1

    if aces_counter_h > 0:

        if total_h > 10:

            total_h += aces_counter_h

        elif aces_counter_h > 1:

            if total_h + 11 + (aces_counter_h-1) <= 21:

                total_h += (11 + (aces_counter_h-1))

            else:

                total_h += aces_counter_h

        else:

            total_h += 11

    return total_h

    

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

    total_h1 = blackjack_hand(hand_1)

    total_h2 = blackjack_hand(hand_2)            

    return True if total_h1 <= 21 and (total_h2 < total_h1 or total_h2 > 21) else False

    



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
### Function from day 6 exercise

def get_max_from_dict(dict):

    max_value = 0

    max_key = 0

    for key, value in dict.items():

        if value > max_value:

            max_value = value

            max_key = key

    return max_key, max_value



def conditional_roulette_probs(history):

    unique_values = []

    history_temp = history[:(len(history) - 1)]  # not consider the last value

    [unique_values.append(value) for value in history_temp if value not in unique_values]

    results = {value: {} for value in unique_values}

    for i in unique_values:

        counter = {}

        for j in range(1, len(history)):

            if history[j - 1] == i:

                if history[j] in counter.keys():

                    counter[history[j]] = counter[history[j]] + 1

                else:

                    counter[history[j]] = 1

        suma = sum(counter.values())

        results[i] = {value: counter[value] / suma for value in counter.keys()}

    return results

        

def my_agent(wheel):

    history = []

    last_number = 0

    while wheel.num_remaining_spins() > 0:

        history.append(last_number)

        if len(history) >= 2:

            probs = conditional_roulette_probs(history)

            if last_number in probs.keys():

                #print(probs[last_number])

                guess = get_max_from_dict(probs[last_number])[0]

                #print(last_number)

            else:

                guess = None

        else:

            guess = None

        last_number = wheel.spin(number_to_bet_on=guess)

            

#help(roulette.evaluate_roulette_strategy)

roulette.evaluate_roulette_strategy(my_agent)