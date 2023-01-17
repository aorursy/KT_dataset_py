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
print(has_lucky_number([1,15,25,6]))
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    a=[]
    for num in nums:
        if num%7==0:
            a.append(True)
        else:
            a.append(False)
    if True in a:
        return True
    else:
        return False
print(has_lucky_number([2,7,2]))
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
    pass
    a=[]
    for num in L:
        if num>thresh:
            a.append(True)
        else:
            a.append(False)
    return a

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    pass
    meals1=[]
    for i in range(len(meals)-1):
        if(meals[i]==meals[i+1]):
            meals1.append(True)
        else:
            meals1.append(False)
    return any(meals1)
print(menu_is_boring(['Spam', 'Eggs', 'Bacon', 'Spam']))
q3.check()
#q3.hint()
#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average yield collected
    """
    pass
    casino=[]
    for i in range(n_runs):
        casino.append(play_slot_machine())
    return (sum(casino)/len(casino))-1
print(estimate_average_slot_payout(1000))
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
    pass
   

#q5.check()
q5.solution()