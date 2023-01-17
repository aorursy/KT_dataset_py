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
         
    return False 
nums1 = [4,8,14,8,7]
cc = has_lucky_number(nums1)
cc
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    lucky = 0
    for num in nums:
        if num % 7 == 0:
            lucky = lucky +1 
        else:
            None
    return True if lucky > 0 else False       
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
    grt = []
    for num in L:
        if num > thresh:
           grt.append(True)
        else:
           grt.append(False)
    return grt         

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals)):
       if i > 0:
          if meals[i] == meals[i-1]: 
             return True
    
    return False   
q3.check()
meals = ['Spam', 'Spam','Eggs', 'Bacon', 'Spam']
lst = []
for i in range(len(meals)):
    if i > 0:
       print (">0",meals[i] )
       if meals[i] == meals[i-1]: 
          print ("Matched",meals[i] )
    
     


#q3.hint()
#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average payout collected
    """
    slot = []
    for row in range(n_runs):
        cost = play_slot_machine()
        slot.append(cost)
    return sum(slot)/ n_runs   

avrage = estimate_average_slot_payout(10000)
avrage
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
    fail = 0
    for simu in range(n_simulations):
        win_cash = start_balance ;
        for spin in range(n_spins):
            if win_cash >= 1:
               #print('win_cash ',win_cash) 
               win_cash = win_cash + play_slot_machine()

            else:
              fail = fail + 1
              #print('cash_out ',spin)
              break
            win_cash = win_cash - 1 # cost of each spin 1$     

    return ((n_simulations - fail)/n_simulations)
q5.check()
n_simulations = 1000
start_balance = 10
n_spins = 10
cash_out =[]
fail = 0
for simu in range(n_simulations):
    win_cash = start_balance ;
    for spin in range(n_spins):
        if win_cash >= 1:
           #print('win_cash ',win_cash) 
           win_cash = win_cash + play_slot_machine()
    
        else:
          fail = fail + 1
          #print('cash_out ',spin)
          break
        win_cash = win_cash - 1 # cost of each spin 1$     

print ((n_simulations - fail)/n_simulations)
#q5.solution()