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
    for num in nums:
        #print(num)
        if num % 7 == 0:
            return True
        
    return False

q1.check()
#q1.hint()
#q1.solution()
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    pass
    #return [n > thresh for n in L]

    y = []
    for n in L:
            y.append(n > thresh)
    return y

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    pass
    for i in range(len(meals)-1):
        if meals[i] == meals[i+1]:
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
    j = 0
    for i in range(n_runs):
        n = play_slot_machine()
        j = j + n - 1
        #print(n)
        z = j/n_runs
    return z
    pass
estimate_average_slot_payout(10000000)
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
    xux = 0
    for i in range(n_simulations):
        s = start_balance
        for t in range(n_spins):
            n = play_slot_machine()
            s = s + n - 1
            if s>=1 and t == n_spins-1:
                xux = xux + 1
    return xux/n_simulations
    
    #successes = 0
    #for _ in range(n_simulations):
    #    balance = start_balance
    #    spins_left = n_spins
    #    while balance >= 1 and spins_left:
    #        # subtract the cost of playing
    #        balance -= 1
    #        balance += play_slot_machine()
    #        spins_left -= 1
    #    # did we make it to the end?
    #    if spins_left == 0:
    #        successes += 1
    #return successes / n_simulations
            

q5.check()
q5.solution()