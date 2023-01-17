# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex5 import *
print('Setup complete.')
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
print(product)
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    lucky = False
    for num in nums:
        if num % 7 == 0 and num != 0:
            lucky = True
    return lucky
        
        
alist = range(12,22)
print(list(alist))
has_lucky_number(alist)
help(any)
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    lucky = False
    for num in nums:
        if num % 7 == 0 and num != 0:
            lucky = True
    return lucky
        
        
alist = range(12,22)
print(list(alist))
has_lucky_number(alist)
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
    bigger = []
#     for num in L:
#         if num > thresh:
#             bigger.append(True)
#         else:
#             bigger.append(False)
#     return bigger
    return [ele > thresh for ele in L]

numlist = [1, 2, 3, 4]
checknum = 2
print(elementwise_greater_than(numlist, checknum))
numlist = range(22)
checknum = 11
print(elementwise_greater_than(numlist, checknum))


q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals)):
        if (i+1)<(len(meals)) and meals[i] == meals[i+1]:
            return True
    
    return False

menu=["Pizza", "Steak", "Chicken", "Meatloaf", "Pizza", "Pizza","Tacos"]
print(menu_is_boring(menu))        
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
    profit = 0
    for i in range(n_runs):
        profit = profit -1 + play_slot_machine()
    return profit/n_runs
    

estimate_average_slot_payout(120000)

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
    if n_spins<=start_balance:
        return 1.0
    else:
        passcount = 0
        for i in range(n_simulations):
            end_balance = start_balance
            spinsleft = n_spins
            while spinsleft > 0 and end_balance > 0: 
                end_balance = end_balance - 1 + play_slot_machine()
                spinsleft -= 1
                
            #print("end_balance is", end_balance)
            if not spinsleft :
                passcount += 1
        print("Out of", n_simulations,",", passcount, "simulations allowed for", n_spins, "spins with ", start_balance, "dollars")    
        return passcount/n_simulations
                
print(slots_survival_probability(6.00,10,5))              
print(slots_survival_probability(10.00,10,1000))
print(slots_survival_probability(25, 150, 10000))
print(slots_survival_probability(1.00, 2, 10000))
print(slots_survival_probability(5.00, 7, 100000))
print(slots_survival_probability(5.00, 10, 100000))
print(slots_survival_probability(5.00, 30, 100000))
q5.check()
q5.solution()