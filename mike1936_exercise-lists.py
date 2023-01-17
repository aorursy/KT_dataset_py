# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex4 import *
print('Setup complete.')
def select_second(L):
    """Return the second element of the given list. If the list has no second
    element, return None.
    """
    try:
        if not L[1] == '':
            return L[1]
        else:
            return None
    except IndexError:
        return None
q1.check()
#q1.hint()
#q1.solution()
def losing_team_captain(teams):
    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)
    from the last listed team
    """
    return teams[-1:][0][1]
    # Slices share the same type of the origin,while element picker picks the element

q2.check()
#q2.hint()
#q2.solution()
def purple_shell(racers):
    """Given a list of racers, set the first place racer (at the front of the list) to last
    place and vice versa.
    
    >>> r = ["Mario", "Bowser", "Luigi"]
    >>> purple_shell(r)
    >>> r
    ["Luigi", "Bowser", "Mario"]
    """
    temp = racers[0]
    racers[0] = racers[-1]
    racers[-1] = temp
    
    ''' NOTES:
    1. By list method: (seems like list is just a pointer-like thing)
    CAN NOT assign to change list directly in a function
    >>> type(a)
    <class 'list'>
    >>> def change(a):
    ...     a = a[::-1]
    ... a = [1, 2, 3]
    ... change(a)
    ... a
    ... [1, 2, 3]
    
    2. By element method:
    Can assign to change int (element of a list) directly in a function:
    
    **** even the element of the list is a list, can change directly
    like in this excercise
    '''

q3.check()
#q3.hint()
#q3.solution()
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# Put your predictions in the list below. Lengths should contain 4 numbers, the
# first being the length of a, the second being the length of b and so on.
lengths = [3,2,0,2]

q4.check()
# line below provides some explanation
#q4.solution()
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    return arrivals.index(name)+1 > len(arrivals)/2 and (not arrivals.index(name) == len(arrivals))

## NOTE: The answer should be wrong, the index has difference '-1'
##       BETWEEN the 'Ture Index' (started from 1)
##       AND     the 'Fake Index' (started from 0) of the element in the array 
##       for python language.

q5.check()
#q5.hint()
#q5.solution()
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    nums.append(0)
    nums.sort()
    return nums.index(0)

###############################################
### SIX COMPARISON METHOD EQ,NE,LT,LE,GT,GE ###
###############################################
# count equals x
def count_eq(nums, x):
    return nums.count(x)
# count not euqals x
def count_ne(nums, x):
    return len(nums) - nums.count(x)
# count less than x
def count_lt(nums, x):
    nums.append(x)
    nums.sort()
    return nums.index(x)
# count less than or equals x
def count_le(nums, x):
    nums.append(x)
    nums.sort()
    return nums.index(x)+nums.count(x)-1
# count greater than x
def count_gt(nums, x):
    nums.append(x)
    nums.sort()
    return nums[::-1].index(x)
# count greater than or equals x
def count_ge(nums, x):
    nums.append(x)
    nums.sort()
    return nums[::-1].index(x)+nums.count(x)-1
q6.check()
#q6.hint()
#q6.solution()