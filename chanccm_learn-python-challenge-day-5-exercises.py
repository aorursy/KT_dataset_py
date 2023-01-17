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
    list=[]
    for i, value in enumerate(L):
        if value > thresh:
            list.append(True)
        else:
            list.append(False)
    return list
    

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals)-1):
        if meals[i] == meals[i+1]:
            return True
    return False

q3.check()
#q3.hint()
#q3.solution()

import inspect
lines = inspect.getsource(play_slot_machine)
print(lines)
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average net profit per run.
    Example calls (note that return value is nondeterministic!):
    >>> estimate_average_slot_payout(1)
    -1
    >>> estimate_average_slot_payout(1)
    0.5
    """
    gain = 0
    for n in range(n_runs):
        gain = gain + play_slot_machine()
    profit = (gain - n_runs)/n_runs
    return profit
   
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
    
    counter = 0
    
    for n in range(n_simulations):
        gain = 0
        times = n_spins
        money = start_balance
        time_counter=0
        #print('start_balance: ', start_balance)
        while ((gain+money) >= 1) and (time_counter <= n_spins):
            gain = gain + play_slot_machine()
            money += -1
            times += -1
            time_counter += 1
            #print('gain', gain, 'money', money, 'time_counter', time_counter, sep=' = ', end='; ')
            
        if (time_counter >= n_spins):
            counter += 1
        else:
            counter += 0
        #print('counter',counter)
        
    probability = counter/n_simulations
    return probability

q5.check()
q5.solution()