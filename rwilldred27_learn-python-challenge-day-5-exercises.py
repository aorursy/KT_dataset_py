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
        print('element: ' + str(num)) # for debugging original code above.
        if num % 7 == 0:
            return True
    # if none of the num in nums divisible by 7
    return False

# reminder to self from Lesson 2: "return is another keyword uniquely associated with functions. 
# When Python encounters a return statement, it exits the function immediately, and passes the 
# value on the right hand side to the calling context"
# note: the bug is that the loop currently only runs on the first element of the list. 
has_lucky_number([5,10,15,20,25,30,40])
#q1.check()
#q1.hint()
#q1.solution()
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    # list comprehension = [output expression + iterator + iterable + conditions]
    return [num > thresh for num in L]

q2.check()
#q2.solution()
# My original version which works. 
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    prior_meal = ''
    for meal in meals:
        print('today\'s meal is: ' + meal) # logging for checking iterations
        if meal == prior_meal:
            return True
        prior_meal = meal # update prior meal to check against next meal
        
    return False


#lunch_menu = ['Hot Dog', 'Chicken', 'Square Pizza', 'Square Pizza', 'Broccoli Sandwich']   
#menu_is_boring(lunch_menu)
q3.check()
# my Version 2 given the solution
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals)-1):
        print('index:' + str(i) + ', value: ' + str(meals[i])) # temporary logging for checks
        if meals[i] == meals[i + 1]: # i is local scope, but meals list is in global scope so this works
            return True
    return False
        
lunch_menu = ['Hot Dog', 'Chicken', 'Square Pizza', 'Square Pizza', 'Broccoli Sandwich']   
menu_is_boring(lunch_menu)
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
    payout = [] # initialize per game payout list

    for i in range(n_runs):
        payout.append(play_slot_machine() - 1) # earnings - $1 cost per play
    
    return sum(payout) / len(payout)

# simulate 10 million pulls to get long run EV
estimate_average_slot_payout(10000000)
    
q4.solution()
# NOTE: earlier versions of my function were running forever, freezing the kernel.
# that's because I was not checking if I made it to at least n_spins. If I won
# a decent amount of money, my balance would balloon and my while loop
# was only checking if the balance >= 1. I would continue each simulation
# until I ran out of money. my solution below ensures that each simulation
# never runs more than the n_spins per simulation, making it manageable for compute. 

def slots_survival_probability(start_balance, n_spins, n_simulations):
    """Return the approximate probability (as a number between 0 and 1) that we can complete the
    given number of spins of the slot machine before running out of money, assuming we start
    with the given balance. Estimate the probability by running the scenario the specified number of times.

    >>> slots_survival_probability(10.00, 10, 1000)
    1.0
    >>> slots_survival_probability(1.00, 2, 1000)
    .25
    """
    # were we able to make it to n_spins, each simulation?
    sim_results = []

    # run each simulation
    for i in range(n_simulations):

        # these 2 variables reset each i'th simulation
        balance = start_balance
        remaining_spins = n_spins
        
        # will run until False
        while balance >= 1 and remaining_spins > 0:
            # update balance with winnings minus $1 cost per spin
            balance += play_slot_machine() - 1
            remaining_spins -= 1
            # LOGGING/TESTING print('sim:' + str(i) + ', sim balance: ' + str(balance) + ', sim completed spins: ' + str(completed_spins))

        # once loop terminates,  measure if we made more spins than n_spins.
        sim_results.append(remaining_spins == 0)

    # empirical probability
    return sum(sim_results) / n_simulations

slots_survival_probability(start_balance=10, n_spins=20, n_simulations=1000)

q5.check()
#q5.solution()