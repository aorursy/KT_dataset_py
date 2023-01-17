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

        else:

            return False

        def has_lucky_number(nums):

            return any([num % 7 == 0 for num in nums])
#q1.hint()

#q1.solution()
# [1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    res = []

    for ele in L:

        res.append(ele > thresh)

    return res



def elementwise_greater_than(L, thresh):

    return [ele > thresh for ele in L]

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

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    res = 0

    for i in range(n_runs):

        res += (play_slot_machine() - 1.0)

    return res / n_runs



estimate_average_slot_payout(1000000)
#q4.solution()