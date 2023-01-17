from learntools.core import binder; binder.bind(globals())

from learntools.python.ex5 import *

print('Setup complete.')
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    for num in nums:

        #print(num)

        if num % 7 == 0:

            return True

    return False

        

has_lucky_number([1,2,3,4,5,6])
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    d = []

    for num in nums:

        if num % 7 == 0:

            d.append(num)

    if len(d) == 0:

        return False

    else:

        return True



q1.check()
#q1.hint()

#q1.solution()

help(any)
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    return [i > thresh for i in L]

        

    pass



q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    

    for i, meal in enumerate(meals[:-1]):

        #print(meal)

        if meals[i] == meals[i+1]:

            return True

    return False

    pass





q3.check()
meals = ['Spam', 'Eggs', 'Spam', 'Spam', 'Bacon', 'Spam']

#print(enumerate(meals[:-1]))
a = [1,2,3,4]

print()
#q3.hint()

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

    play_slot_machine()

    

    pass
q4.solution()