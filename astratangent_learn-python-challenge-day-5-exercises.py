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
    ind = 0
    if len(nums) > 0:
        for num in nums:
            if num % 7 == 0:
                ind += 1
        return bool(ind)    
               
    else:
        return False

q1.check()
#help(any)
#q1.hint()
#q1.solution()
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    return [Li > thresh for Li in L]

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    i = 0
    for ele1 in meals:
        #print(ele1)
        if ele1 != 'Spam':
            print(ele1)
            for ele2 in meals[i+1:]:
                if ele1 == ele2:
                    return True
        i +=1
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
    res=0
    for n in range(n_runs):
        res+=play_slot_machine()-1
    return res/n_runs
estimate_average_slot_payout(5)
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
    if start_balance >= n_spins:
        return 1.0
    
    spins_left= n_spins
    balance   = start_balance
    successes = 0
    failures  = 0
    i = 0
    while i < n_simulations:
        balance += -1 + play_slot_machine()
        #print ('balance = ', balance)
        i += 1
        if balance > 0 and spins_left > 0:
            spins_left -=1
            if spins_left <= 0:
                successes +=1
                spins_left = n_spins
                balance   = start_balance
        elif balance <= 0 and spins_left > 0:
            failures += 1
            spins_left = n_spins
            balance   = start_balance  
    #print('successes =', successes, ' failures = ', failures)            
    return successes / (successes + failures)       

slots_survival_probability(1,2,10000)
q5.check()
#q5.solution()