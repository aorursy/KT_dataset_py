from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    return L[1] if len(L) >1 else None

    pass



q1.check()
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    return None if len(L) == 1 else L[1]

select_second([1])
#q1.hint()

#q1.solution()
def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    return teams[-1][1]

    pass



q2.check()
#q2.hint()

q2.solution()
def purple_shell(racers):

    """Given a list of racers, set the first place racer (at the front of the list) to last

    place and vice versa.

    

    >>> r = ["Mario", "Bowser", "Luigi"]

    >>> purple_shell(r)

    >>> r

    ["Luigi", "Bowser", "Mario"]

    """

    buffer = [racers[0], racers[-1]]    

    racers[0] = buffer[1]

    racers[-1] = buffer[0]

    #return racers

    pass



q3.check()
a=[1,2,3,4,5]

print(a.reverse)
#q3.hint()

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

#q4.solution()
def fashionably_late(arrivals, name):

    """Given an ordered list of arrivals to the party and a name, return whether the guest with that

    name was fashionably late.

    """

    if (len(arrivals)%2) == 0:

        return name in  arrivals[(int(len(arrivals)/2)):-1]

    else:

        return name in  arrivals[((int(len(arrivals)/2))+1):-1]

    pass



q5.check()
import math

a=[1,2,3,4,5]

b=a[1:]

#print(2 in b)

#math.floor(3.9)

#int(3.999)

int(len(a)/2)
#q5.hint()

q5.solution()