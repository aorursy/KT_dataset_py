from learntools.core import binder; binder.bind(globals())
from learntools.python.ex5 import *
print('Setup complete.')
array = [2,5,6,7,3,4]

def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    
    for num in nums:
        print("1")
        if num % 7 == 0:
            return True
        else:
            return False
        
has_lucky_number(array)
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    for num in nums:
        if num % 7 == 0:
            return True
    return False
        

# Check your answer
q1.check()
q1.hint()
#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    new_list = []
    for i in L:
        if (i > thresh): 
            new_list.append(True)
        else:
            new_list.append(False)
    return new_list
    

# Check your answer
q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    previous_meal = ""
    for meal in meals:
        if (meal == previous_meal):
            return True
        else:
            previous_meal = meal
    return False

# Check your answer
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
    i=0
    payouts = []
    while i <= n_runs:
        payouts.append(play_slot_machine())
        i += 1
    average = sum(payouts)/len(payouts)
    print(average)
    return average

estimate_average_slot_payout(500)
# Check your answer (Run this code cell to receive credit!)
q4.solution()