# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex4 import *
print('Setup complete.')
def select_second(L):
    """Return the second element of the given list. If the list has no second
    element, return None.
    """
    items = len(L)
    if items < 2:
        return
    else:
        return L[1]
q1.check()
#q1.hint()
#q1.solution()
def losing_team_captain(teams):
    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)
    from the last listed team
    """
    worst_team = teams[(len(teams) -1)]
    worst_captain = worst_team [1]
    return worst_captain
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
    last = racers.pop()
    first = racers[0]
    racers.remove(first)
    top = racers.insert(0, last)
    racers.append(first)

q3.check()
#q3.hint()
#q3.solution()
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# Put your predictions in the list below. Lengths should contain 4 numbers, the
# first being the length of a, the second being the length of b and so on.
lengths = [3, 2, 0, 2]

q4.check()
# line below provides some explanation
#q4.solution()
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    guests = len(arrivals)
    half = (len(arrivals)/ 2) 
    arrival_position = arrivals.index(name)
    if arrival_position >= half and arrival_position != (guests -1):
        fash_late = True
    else:
        fash_late = False
    return fash_late
q5.check()
#q5.hint()
#q5.solution()
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    nums.append(0)
    rank = sorted(nums)
    pos = rank.index(0)
    return pos

q6.check()
#q6.hint()
#q6.solution()