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

        if (num % 7 == 0):

            return True



    return False





    pass

# Check your answer

q1.check()
q1.hint()

q1.solution()
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    li = []

    for i in L:

        li.append(i>thresh)

    

    return li

    pass



# Check your answer

q2.check()
q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    li=[]

    for i in range(len(meals)-1):

        if meals[i] == meals[i+1]:

            return True

        

    return False

    pass



# Check your answer

q3.check()
q3.hint()

q3.solution()
play_slot_machine()
import random
def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    cost = 1

    returns = random.choice([0,random.randint(1,1000000)])

    

    payout = 0

    for i in range(n_runs):

        payout =  payout + (returns - cost)

    return payout

    pass
random.choice([0,random.randint(1,1000000)])
estimate_average_slot_payout(1000000)
# Check your answer (Run this code cell to receive credit!)

q4.solution()