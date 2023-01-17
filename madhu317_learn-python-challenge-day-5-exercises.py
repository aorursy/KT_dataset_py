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
        #print ("loop")
        if num % 7 == 0:
            return True
            
       
    return False

    
            

#nums = [1,2,3,4,5,6,7,8]
#has_lucky_number(nums)
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
    a = []
    for elements in L: 
        a.append(elements > thresh)
    return a
#L=[1,2,3,4,5]
#thresh = 2
#elementwise_greater_than(L, thresh)
q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for m in range(len (meals) -1):
        if meals[m] == meals[m+ 1]:
            return True
    return False 
        

q3.check()
#q3.hint()
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
    a = []
    for i in range(n_runs):
        a.append(play_slot_machine() -1)
    average = sum(a)/n_runs
    return average
estimate_average_slot_payout(100)
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
    prob = 0
    # A convention in Python is to use '_' to name variables we won't use
    for i in range(n_simulations):
        RemainingBalance = start_balance
        RemainingSpins = n_spins
        while RemainingBalance >= 1 and RemainingSpins:
            # subtract the cost of playing
            RemainingBalance -= 1
            RemainingBalance += play_slot_machine()
            RemainingSpins -= 1
        # did we make it to the end?
        if RemainingSpins == 0:
            prob += 1
    return prob / n_simulations

q5.check()
#q5.solution()