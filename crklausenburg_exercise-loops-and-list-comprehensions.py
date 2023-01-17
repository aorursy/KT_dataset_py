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

    if len(nums) == 0:

        return False

    

    to_return = False

    

    for num in nums:

        if num % 7 == 0:

            to_return = True

            break

        else:

            to_return = False

        

    return to_return



q1.check()
q1.hint()

q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    to_return = []

    

    for x in L:

        to_return.append(x > thresh)

    return to_return



q2.check()
q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    return any([meals[i] == meals[i-1] for i in range(1, len(meals))])



q3.check()
q3.hint()

q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    sum = 0

    for run in range(n_runs):

        sum = sum + (play_slot_machine() - 1)

    

    return sum / n_runs



estimate_average_slot_payout(5000000)
q4.solution()