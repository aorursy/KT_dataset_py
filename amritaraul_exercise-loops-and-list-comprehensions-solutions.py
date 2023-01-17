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

    #Solution1

    return any ([num%7==0 for num in nums])

    #Solution2

    for num in nums:

        print

        if num % 7 == 0:

            return True

    return False



# Check your answer

q1.check()
q1.hint()

q1.solution()
[1, 2, 3, 4]> 2
def elementwise_greater_than(L, thresh):

    #Solution1

    return([n>thresh for n in L])

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    #Solution2

    list=[]

    for n in L:

        return list.append(n>thresh)    

    return list

    

# Check your answer

q2.check()
q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    return any(x == y for x,y in zip(meals, meals[1:]))

    for dish in range(len(meals)-1):

        if meals[dish]==meals[dish+1]:

            return True

    return False

# Check your answer

q3.check()
q3.hint()

q3.solution()
play_slot_machine()
#Run the slot machine n_runs times and return the average net profit per run.

#Solution1

def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average payout collected

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    winnings = 0

    for i in range(n_runs):

        winnings += play_slot_machine()-1



    return (winnings/n_runs)                      

    

test_runs = 10000000

print("Estimating for", test_runs, "runs...")

print(estimate_average_slot_payout(test_runs))
#Solution2

def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average payout collected

    """

    return sum([play_slot_machine() - 1 for i in range(n_runs)]) / n_runs

estimate_average_slot_payout(100000000)
# Check your answer (Run this code cell to receive credit!)

q4.solution()