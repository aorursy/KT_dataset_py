from learntools.core import binder; binder.bind(globals())

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

    graph.set_ylim(0,400)

    graph.set_ylabel("Balance")



graph = jimmy_slots.get_graph()

prettify_graph(graph)

graph
def prettify_graph(graph):

    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis

    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks

    as dollar amounts using the "$" symbol.)

    """

    graph.set_title("Results of 500 slot machine pulls")

    # Complete steps 2 and 3 here

    graph.set_ylim(0,400)

    graph.set_ylabel("Balance")

    yticks = [0,100,200,300,400]

    graph.set_yticks(yticks)

    graph.set_yticklabels([r"\$" + str(n) for n in yticks])



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
#q2.hint()
#q2.solution()
def total_count(hand):

    total = 0

    aces = 0

    

    for i in hand:

        if i in ['J','Q','K']:

            total += 10

        elif i == 'A':

            aces += 1

        else:

            total += int(i)

    total += aces

    

    while total + 10 <= 21 and aces > 0:

        total += 10

        aces -= 1

        

    return total

    

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

    total_1 = total_count(hand_1)

    total_2 = total_count(hand_2)

    

    return total_1 <= 21 and (total_1 > total_2 or total_2 > 21)

    pass



q3.check()
#q3.hint()

#q3.solution()