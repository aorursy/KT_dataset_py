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
    if not nums:
        return False
    else:
        ret_val = False
        for num in nums:
            print("num:", num)
            if (num % 7 == 0) and (num != 0):
                ret_val =  True
        return ret_val

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
    #print("L:", L)
    #ret_list = [True for val in L if val > thresh else False]
    ret_list = []
    for ele in L:
        #if val > thresh:
        #    ret_list.append(True)
        #else:
        #    ret_list.append(False)
        ret_list.append(ele > thresh)
    #print("ret_list", ret_list)       
    return ret_list

q2.check()
q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    #pass
    len_meals = len(meals)
    for i in range(len_meals):
        if i > 0 and i < len_meals:
            print("meals[i]", meals[i], "meals[i-1]", meals[i-1])
            if meals[i] == meals[i-1]:
                return True
    return False

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
    print("n_runs", n_runs)
    amt_played = 0
    amt_won = 0
    for i in range(n_runs):
        amt_played = amt_played + 1
        amt_won = amt_won + play_slot_machine()
    print("amt_played:", amt_played, "amt_won:", amt_won)
    return((amt_won - amt_played) / n_runs)

estimate_average_slot_payout(1000000)
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
    print("start balance: ", start_balance, "n_spins: ", n_spins, "n_simulations: ", n_simulations)
    num_sims_survived = 0
    if n_simulations >= 1:
        for sim in range(n_simulations):
                curr_balance = start_balance
                curr_spins = 0
                curr_sim_done = False
                while (curr_balance >=1 ) and (not curr_sim_done) and (curr_spins != n_spins):
                    curr_balance = curr_balance - 1 + play_slot_machine()
                    curr_spins = curr_spins + 1
                    if curr_spins == n_spins:
                        num_sims_survived = num_sims_survived + 1
                        curr_sim_done = True
        return(num_sims_survived / n_simulations)

q5.check()
q5.solution()