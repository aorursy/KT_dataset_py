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

# help (any)

    return any ([num % 7 == 0 for num in nums])

#             return True

#     return False







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

    return [l > thresh for l in L]

#     pass



q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    no_of_meal_days = len(meals)

    if no_of_meal_days > 1:

        for i in range(no_of_meal_days-1):

              if meals[i]==meals[i+1]:

                return True

    return False

            

        

# help (while)

#     pass



q3.check()
q3.hint()

q3.solution()
play_slot_machine()
# import numpy as np

def estimate_average_slot_payout(n_runs):

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

estimate_average_slot_payout(1000000000)

#     random_payout = np.random.rand(n_runs)*1.5 - 1

#     return np.average(random_payout)



# q4.hint()

# help (average)

#     pass

q4.solution()