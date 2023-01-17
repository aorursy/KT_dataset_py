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
import matplotlib as plt

def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    graph.set_ylim(bottom=0)
    graph.set_ylabel("Balance")
    
    formatter = plt.ticker.FormatStrFormatter('$%i')
    graph.yaxis.set_major_formatter(formatter)

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
q2.solution()
def aces_in_hand(hand):
    return hand.count('A')

def hand_total(hand):
    s = 0
    for c in hand:
        if c.isdigit():
            s += int(c)
        elif (c in ['J', 'Q', 'K']):
            s += 10
        elif c == 'A':
            s += 11
    a = aces_in_hand(hand)
    while (a > 0 and s > 21):
        s -= 10
        a -= 1
    return s

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
    t1 = hand_total(hand_1)
    t2 = hand_total(hand_2)
    return t1 <= 21 and (t1 > t2 or t2 > 21)

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
def max_roulette_probs_pair(history):
    """Return a pair (v, w), such that after v appears,
    w appears with much higher probability than other numbers

    Example: 
    max_roulette_probs_pair([1, 3, 1, 5, 1])
    > 3, 1
    """
    counts = {}
    totals = {}
    for i, v in enumerate(history[:-1]):
        counts.setdefault(v, {})
        totals.setdefault(v, 0)
        w = history[i + 1]
        counts[v].setdefault(w, 0)
        counts[v][w] += 1
        totals[v] += 1
    #print("Counts: ", counts)
    #print("Totals: ", totals)
    
    probs = {}
    for v in counts:
        probs[v] = (-1, 0.0)
        for w in counts[v]:
            w_prob = counts[v][w] / totals[v]
            if (w_prob > probs[v][1]):
                probs[v] = (w, w_prob)
    #print("Probs: ", probs)

    v_best, w_best, max_prob = -1, -1, 0.0
    for v, (w, prob) in probs.items():
        if prob > max_prob:
            v_best, w_best, max_prob = v, w, prob
    #print("Result: ", v_best, w_best, max_prob)
    
    return v_best, w_best

def my_agent(wheel):
    total_spins = wheel.num_remaining_spins()
    history = []
    for i in range(2 * total_spins // 3):
        history.append(wheel.spin())
        
    got, bet = max_roulette_probs_pair(history)
    
    last_number = -1
    while wheel.num_remaining_spins() > 0:
        if last_number == got:
            guess = bet
        else:
            guess = None
        last_number = wheel.spin(number_to_bet_on=guess)
    
roulette.evaluate_roulette_strategy(my_agent)
help(roulette.evaluate_roulette_strategy)
class History:
    def __init__(self, size, debug=False):
        if debug:
            print("INIT")
        self.wheel_size = size
        self.last = -1
        self.totals = [0 for v in range(size)]
        self.counts = [[0 for w in range(size)] for v in range(size)]
        self.probs = [[0.0 for w in range(size)] for v in range(size)]
        if debug:
            self.print_stat()
        
    def update(self, v, debug=False):
        if debug:
            print("UPDATE", v)
        last = self.last
        if last != -1:
            self.totals[last] += 1
            self.counts[last][v] += 1
            for w in range(self.wheel_size):
                self.probs[last][w] = self.counts[last][w] / self.totals[last]
        self.last = v
        if debug:
            self.print_stat()
        
    def get_bet(self, v, debug=False):
        if debug:
            print("GET_BET", v)
        probs = self.probs[v]
        bet = probs.index(max(probs))
        if debug:
            print("Bet:", bet)
        return bet
    
    def highest_prob_pair(self, debug=False):
        if debug:
            print("HIGHEST_PROB_PAIR")
        v_best, w_best, max_prob = -1, -1, 0.0
        for v, w_probs in enumerate(self.probs):
            for w, prob in enumerate(w_probs):
                if prob > max_prob:
                    v_best, w_best, max_prob = v, w, prob
        if debug:
            print("The pair:", v_best, w_best)
        return v_best, w_best
    
    def print_stat(self):
        print("Last value:", self.last)
        print("Totals:", self.totals)
        print("Counts:", self.counts)
        print("Probs:", self.probs)

Debug = False
Wheel_size = 11
        
def my_better_agent(wheel):
    history = History(Wheel_size, debug=Debug)
    total_spins = wheel.num_remaining_spins()
    #start_bets = total_spins // 8
    #trust_stats = 7 * total_spins // 8
    start_bets = 0
    trust_stats = 300
    
    guess = None
    
    for i in range(start_bets):
        v = wheel.spin()
        history.update(v, Debug)
    
    for i in range(start_bets, trust_stats):
        v = wheel.spin(number_to_bet_on=guess)
        history.update(v, Debug)
        guess = history.get_bet(v, Debug)
        
    v_best, w_best = history.highest_prob_pair(debug=Debug)
    guess = None
    while wheel.num_remaining_spins() > 0:
        if v == v_best:
            guess = w_best
        else:
            guess = None
        v = wheel.spin(number_to_bet_on=guess)

#roulette.evaluate_roulette_strategy(my_better_agent, wheel_size=Wheel_size)
roulette.evaluate_roulette_strategy(my_better_agent, wheel_size=Wheel_size, num_spins_per_simulation=1000)