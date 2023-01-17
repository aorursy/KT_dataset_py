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

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    for num in nums:

        if num % 7 == 0:

            return True

        else:

            return False



# Check your answer

q1.check()
q1.hint()

#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    

    out_l=[]

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """



    for i in L:

        

        out_l.append(i>thresh)

    return out_l



# Check your answer

q2.check()
#q2.solution()
def menu_is_boring(meals):

    for x in (range(len(meals)-1)):

        if (meals[x] == meals[x+1]):

            return True

    return False

        

            

   

        

        

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

   



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):

    a

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    

    x = 0

    for i in range(n_runs):

        x += play_slot_machine()-1

    return x/n_runs



 
# Check your answer (Run this code cell to receive credit!)

q4.solution()