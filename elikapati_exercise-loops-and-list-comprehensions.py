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

    if len(nums) <= 0:

        return False



    for num in nums:

        if (num % 7 == 0):

            return True

    return False



q1.check()
#q1.hint()

#q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    result = []

    for n in L:

        outcome = True if (n > thresh) else False

        result.append(outcome)

    return result



q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    l = len(meals)

    if l <= 0:

        return False



    ymeal = meals[0]

    for i in range(1, l):

        cmeal = meals[i]

        if (ymeal == cmeal):

            return True

        else:

            pass

        ymeal = cmeal

    return False



q3.check()
#q3.hint()

#q3.solution()
play_slot_machine()
from random import randint

def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    wins = randint(1, n_runs)

    loss = n_runs - wins

    net = wins * 0.5 - loss * 1

    print('wins =', wins, 'loss =', loss)

    return net
q4.solution()