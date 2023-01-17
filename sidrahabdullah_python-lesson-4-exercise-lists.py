from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    if len(L) < 2:

        return None

    return L[1]



# Check your answer

q1.check()
q1.hint()

q1.solution()
def losing_team_captain(teams):

    return teams[-1][1];

    



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
def purple_shell(racers):

    temp = racers[0]

    racers[0] = racers[-1]

    racers[-1] = temp



# Check your answer

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



# Check your answer

q4.check()
# line below provides some explanation

#q4.solution()
def fashionably_late(arrivals, name):

    order = arrivals.index(name)

    return order >= len(arrivals) / 2 and order != len(arrivals) - 1



# Check your answer

q5.check()
#q5.hint()

#q5.solution()