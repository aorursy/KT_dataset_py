from learntools.core import binder; binder.bind(globals())

from learntools.python.ex5 import *

print('Setup complete.')
def has_lucky_number(nums):

    for num in nums:

        if num % 7 == 0:

            return True

    # We've exhausted the list without finding a lucky number

    return False

def has_lucky_number(nums):

    return any([num % 7 == 0 for num in nums])

# Check your answer

q1.check()
#q1.hint()

#q1.solution()
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    return [ele > thresh for ele in L]



# Check your answer

q2.check()
#q2.solution()
def menu_is_boring(meals):

    # Iterate over all indices of the list, except the last one

    for i in range(len(meals)-1):

        if meals[i] == meals[i+1]:

            return True

    return False



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):

    result = []

    for x in range(1, n_runs):

        result.append(play_slot_machine())

    return (sum(result) - n_runs)/n_runs
# Check your answer (Run this code cell to receive credit!)

q4.solution()