from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    pass

    if len(L)>=2: return L[1]

    else: return None

q1.check()
q1.hint()

q1.solution()
def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    pass

    return teams[-1][1]



q2.check()
q2.hint()

q2.solution()
def purple_shell(racers):

    """Given a list of racers, set the first place racer (at the front of the list) to last

    place and vice versa.

    

    >>> r = ["Mario", "Bowser", "Luigi"]

    >>> purple_shell(r)

    >>> r

    ["Luigi", "Bowser", "Mario"]

    """

    pass

    racers[0],racers[-1]=racers[-1],racers[0]



q3.check()
q3.hint()

q3.solution()
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]



# Put your predictions in the list below. Lengths should contain 4 numbers, the

# first being the length of a, the second being the length of b and so on.

lengths = [3,2,0,2]

q4.check()
# line below provides some explanation

q4.solution()
def fashionably_late(arrivals, name):

    """Given an ordered list of arrivals to the party and a name, return whether the guest with that

    name was fashionably late.

    """

    pass

    order = arrivals.index(name) #returning name from the list 

    return order >= len(arrivals) / 2 and order != len(arrivals) - 1 

#more than half of total arrivals but not equals to last



q5.check()
q5.hint()

q5.solution()
def count_negatives(nums):

    """Return the number of negative numbers in the given list.

    

    >>> count_negatives([5, -1, -2, 0, 3])

    2

    """

    nums.append(0)

    nums.sort()

    return nums.index(0) #since the list is sorted, index of 0 always gives number of negative elements



q6.check()
q6.hint()
q6.solution()