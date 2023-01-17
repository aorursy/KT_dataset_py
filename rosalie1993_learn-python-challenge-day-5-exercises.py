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

    for num in nums:
        if num % 7 == 0:
            return True
    return False

q1.check()
q1.hint()
q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    R = []
    for num in L:
        if num > thresh:
            R.append(True)
        else:
            R.append(False)
    return(R)

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for meal in range(len(meals)-1):
        if meals[meal] == meals[meal+1]:
            return(True)
    return False
q3.check()
q3.hint()
#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average payout collected
    """
    R=[]
    for n in range(n_runs):
        R.append(play_slot_machine()-1)
        #print(R)
    R1 = sum(R)/len(R)
    return(R1)
    pass
q4.check()
estimate_average_slot_payout(100000)
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
    if start_balance >= n_spins:
        return(1.0)
    else:
        profit = estimate_average_slot_payout(n_simulations)
        start_balance += profit
        print(start_balance)
        if start_balance >= n_spins:
            return(1.0)
        else:
            return(profit)
    pass

q5.check()
q5.solution()