# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex5 import *
print('Setup complete.')
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    for num in nums:
        if num % 7 == 0 and num!=0:
            return True
        else:
            return False
x =[x for x in range(20)]
has_lucky_number(x)
x
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    nums.append(1)
    result = False
    for num in nums:
        if num % 7 == 0 :
            result = True
    return result

q1.check()
#q1.hint()
#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    results =[]
    for number in L:
        if number>thresh:
            results.append(True)
        else:
            results.append(False)
    return results

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for index in range(len(meals)-1):
        if meals[index]==meals[index+1]:
            return True
    return False

q3.check()
q3.hint()
#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average net profit per run.
    Example calls (note that return value is nondeterministic!):
    >>> estimate_average_slot_payout(1)
    -1
    >>> estimate_average_slot_payout(1)
    0.5
    """
    money = 0
    for run in range(n_runs):
        gain = play_slot_machine()
        money = money - 1 + gain
    return money

estimate_average_slot_payout(10)
q4.solution()
def game_outcome(start_balance,n_runs):
    nr_games = 0
    balance = start_balance
    for run in range(n_runs):
        if balance>=1:
            nr_games = nr_games+1
            balance = balance - 1 + play_slot_machine()
    return nr_games / float(n_runs)
        
game_outcome(5,10)    
def slots_survival_probability(start_balance, n_spins, n_simulations):
    """Return the approximate probability (as a number between 0 and 1) that we can complete the 
    given number of spins of the slot machine before running out of money, assuming we start 
    with the given balance. Estimate the probability by running the scenario the specified number of times.
    
    >>> slots_survival_probability(10.00, 10, 1000)
    1.0
    >>> slots_survival_probability(1.00, 2, 1000)
    .25
    """
    successes = 0
    for sim in range(n_simulations):
        if game_outcome(start_balance,n_spins)==1:
            successes = successes + 1
    return successes/ n_simulations


print(slots_survival_probability(10,100,100000))
q5.check()
q5.solution()