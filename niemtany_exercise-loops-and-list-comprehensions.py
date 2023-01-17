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
  return any([num % 7 == 0 for num in nums])

q1.check()
#q1.hint()
#q1.solution()
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    res = []
    for ele in L:
        res.append(ele > thresh)
    return res

q2.check()
#q2.solution()
def menu_is_boring(meals):
    for i in range(len(meals)-1):
        if meals[i] == meals[i+1]:
            return True
    return False

q3.check()
#q3.hint()
q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
   
    pass
q4.solution()
def slots_survival_probability(start_balance, n_spins, n_simulations):
    successes = 0
    # A convention in Python is to use '_' to name variables we won't use
    for _ in range(n_simulations):
        balance = start_balance
        spins_left = n_spins
        while balance >= 1 and spins_left:
            # subtract the cost of playing
            balance -= 1
            balance += play_slot_machine()
            spins_left -= 1
        # did we make it to the end?
        if spins_left == 0:
            successes += 1
    return successes / n_simulations

q5.check()
q5.solution()