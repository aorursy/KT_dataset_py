# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
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
    luckyNumber = False
    for num in nums:
        if num % 7 == 0:
            luckyNumber = True
 
    return luckyNumber

q1.check()
help(any)
#q1.hint()
#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
   
    return [l>thresh for l in L]

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    lastMeal = ""
    i = 0
    for i, meal in enumerate(meals):
        if(i>0):
            if(meals[i] == meals[i-1]):
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
    i=1
    while i <= n_runs:
        profit = play_slot_machine() - 1
        i = i + 1
        
    avg_proft = profit/n_runs

    return avg_proft

estimate_average_slot_payout(1)
q4.solution()
def slots_survival_probability(start_balance, n_spins, n_simulations):
    """Return the approximate probability (as a number between 0 and 1) that we can complete the 
    given number of spins of the slot machine before running out of money, assuming we start 
    with the given balance. Estimate the probability by running the scenario the specified number of times.
    
    >>> slots_survival_probability(10.00, 10, 1000)
    1.0
    >>> slots_survival_probability(1.00, 2, 1000)
    .25
    """
    cost = 1
    balance = start_balance
    successes = 0
    
    for i in range(n_simulations):
        spins = 1
        while spins <= n_spins:
            balance = balance + play_slot_machine() - 1
            spins = spins + 1
            if balance < 1:
                break

        if spins > n_spins:
            successes += 1
        
    return successes/n_simulations

slots_survival_probability(10.00, 10, 1000)
#q5.check()
q5.solution()