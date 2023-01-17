from learntools.core import binder; binder.bind(globals())

from learntools.python.ex5 import *

print('Setup complete.')
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    list = []

    i = 0

    for num in nums:

        if num % 7 == 0:

            list.append(True)

        else:

            list.append(False)

    for l in list:

        for i in range((len(list)+1)):

            if l == True:

                i += 1

            else:

                pass

    return True if i>0 else False

        

has_lucky_number([1,2,3,4,5,6,7])
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    list = [True for num in nums 

            if num % 7 == 0]

    i = 0

    '''for num in nums:

        if num % 7 == 0:

            list.append(True)

        else:

            list.append(False)'''

    for l in list:

        if l == True:

            i += 1   

        else:

            pass

    return True if i>0 else False



has_lucky_number([1,2,3,4,5,6,7])

# Check your answer

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

    list = []

    for l in L:

        if l > thresh:

            list.append(True)

        else:

            list.append(False)

    return list



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

    return False



# Check your answer

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

    list = []

    i = 0

    while i < (n_runs-1):

        list[i] = play_slot_machine() 

    print(list[:6])



estimate_average_slot_payout(19000)
# Check your answer (Run this code cell to receive credit!)

q4.solution()