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
#q1.hint()
#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    return [True if(x > thresh) else False for x in L]

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    def recursive_boring_func(meals) :
        if(len(meals) > 1 ) :
            if(meals[0] == meals[1]) :
                return True
            else :
                return recursive_boring_func(meals[1:])
        else :
            return False
        
    return recursive_boring_func(meals)

q3.check()
#q3.hint()
#q3.solution()
play_slot_machine()
import numpy as np
my_np_array = np.array([play_slot_machine() for i in range(10000)] )
print(np.mean(my_np_array))
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average payout collected
    """
    return [play_slot_machine() for i in range(n_runs)]
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
    spin_cost = 1 
    def more_spins_stimation(start_balance, expected_spins) :
        spin_cost = 1
        current_balance = start_balance
        current_spin = 0
        while(True) :
            if(current_balance > 0) :
                current_balance = current_balance + play_slot_machine() - spin_cost
                current_spin += 1
                if(expected_spins <= current_spin) :
                    return True
            else :
                return False
            
    simulation = [more_spins_stimation(start_balance, n_spins ) for i in range(n_simulations)]
    wins = 0
    for result in simulation :
        if(result) :
            wins +=1
    # print(simulation)
    return float(wins/n_simulations)
        
q5.check()
#q5.solution()