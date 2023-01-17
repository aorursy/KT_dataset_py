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

    bools = []

    for num in L:

        num = bool(num > thresh)

        bools.append(num)

    return bools

            



# Check your answer

q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    for i in range(len(meals)-1):

        if meals[i] == meals[i+1]:

            return True

        else:

            i+=1 #not needed

    return False

      

        



# Check your answer

q3.check()
q3.hint()

#q3.solution()
play_slot_machine()
global total

def estimate_average_slot_payout(n_runs):

    total = 0

    for i in range(n_runs):

        total += play_slot_machine() - 1

    return (total / n_runs)



estimate_average_slot_payout(500000)



# def max_payout(n_runs):

#    maximum = 0

#     for i in range(n_runs):

#         if play_slot_machine() > maximum:

#             maximum = play_slot_machine()

#         else:

#             continue

#     return maximum



# max_payout(500)

        



        

        
# Check your answer (Run this code cell to receive credit!)

q4.solution()