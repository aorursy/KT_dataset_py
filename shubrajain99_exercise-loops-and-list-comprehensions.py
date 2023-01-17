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

        
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    s=0

    for num in nums:

        if num % 7 == 0:

            s=1

            break

        else:

            s=0

    if s==1:

        return True

    else:

        return False

        



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    for i in range(len(L)):

        if L[i]>thresh:

            L[i]=True

        else:

            L[i]=False

    return L

 

  

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    pass



# Check your answer

q2.check()
#q2.solution()




def menu_is_boring(meals):

    # Iterate over all indices of the list, except the last one

    for i in range(len(meals)-1):

        if meals[i] == meals[i+1]:

            return True

    return False

 



q3.check()
#q3.hint()

#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):

    sum=0

    for i in n_runs:

        sum =sum+play_slot_machine()

    

    result=sum/n_runs

    

    return result

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

 
# Check your answer (Run this code cell to receive credit)

q4.solution()