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

#             print(num)

            return True

        else:

            continue

    return False



# has_lucky_number([1,2,3,4,5,6,8,8,8,8,8,8,8,8,7])



# Check your answer

q1.check()
q1.hint()

q1.solution()
help(any)
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

#     return [True for l in L if l>thresh] wrong answer

    return [ele > thresh for ele in L]



# Check your answer

q2.check()
q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    for i in range(len(meals)-1): # from [0,len-1)

    # Iterate over all indices of the list, except the last one

            if meals[i] == meals[i+1]:

                return True

    return False

    

    

#     failed attempt : suppose to return boolean,not list

#     result = []

#     temp = meals[0]

#     for i in range(1,len(meals)):      

#         if(meals[i]==temp):

#             result.append(True)

#         temp = meals[i]

#         result.append(False)

#     return result





#     failed attempt: IndexError: list index out of range

#     result = False

#     temp = meals[0]

#     for i in range(1,len(meals)):      

#         if(meals[i]==temp):

#             result = True

#             break

#         temp = meals[i]

#         result = False

#     return result

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

    gain = 0.0

    for i in range(n_runs):

        gain += play_slot_machine()

        gain -= 1  #  each play costs $1

    print(gain/n_runs)

    return gain/n_runs



estimate_average_slot_payout(1000000)
# Check your answer (Run this code cell to receive credit!)

q4.solution()