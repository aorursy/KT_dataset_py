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

       # else:

    return False



q1.check()
#q1.hint()

#q1.solution()

help(any)
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    result=[]

    for i in L: 

        if i>thresh:

            result.append(True)

        else:

            result.append(False)

    #return [ele > thresh for ele in L]

    return result



q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    l=len(meals)

    for i in range(l-1):

          if (meals[i]==meals[i+1]):

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

    t=n_runs

    profit=0

    while (n_runs!=0):

        profit= profit+play_slot_machine()-1

        n_runs=n_runs-1

    return (profit/t)    
estimate_average_slot_payout(1000000)
#q4.solution()