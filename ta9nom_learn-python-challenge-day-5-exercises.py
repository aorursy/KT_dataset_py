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
        if num % 7 == 0:
            return True
        else:
            return False
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    n = 0
    for num in nums:
        if num % 7 == 0:
            n = n + 1
        else:
            pass
    return True if n > 0 else False
        
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
    #pass
    tmp = []
    for x in L:
        if x > thresh:
            tmp.append(True)
        else:
            tmp.append(False)
            
    return tmp

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    #pass
    for i in range(len(meals)-1):
        if meals[i] == meals[i+1]:
            return True
    return False

q3.check()
q3.hint()
q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average net profit per run.
    Example calls (note that return value is nondeterministic!):
    >>> estimate_average_slot_payout(1)
    -1
    >>> estimate_average_slot_payout(1)
    0.5
    """
    #pass
    n = n_runs
    tot = 0
    while n > 0:
        tot = tot - 1 + play_slot_machine()
        n = n - 1
    return tot / n_runs

estimate_average_slot_payout(1000000)
#q4.solution()
def slots_survival_probability(start_balance, n_spins, n_simulations):
    """Return the approximate probability (as a number between 0 and 1) that we can complete the 
    given number of spins of the slot machine before running out of money, assuming we start 
    with the given balance. Estimate the probability by running the scenario the specified number of times.
    
    >>> slots_survival_probability(10.00, 10, 1000)
    1.0
    >>> slots_survival_probability(1.00, 2, 1000)
    .25
    """
    #pass
    
    n_sim = n_simulations
    sur = 0
    while n_sim > 0:
        n = n_spins
        tot = start_balance
        while ((n > 0) and (tot >= 1)):
            tot = tot - 1 + play_slot_machine()
            n = n - 1
        if n == 0:
            sur = sur + 1    
        n_sim = n_sim - 1
    return sur / n_simulations

q5.check()
q5.solution()