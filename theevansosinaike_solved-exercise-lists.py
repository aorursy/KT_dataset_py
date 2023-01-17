from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    if len(L) > 1:

        return L[1]

    else:

        return None

q1.check()
#q1.hint()

#q1.solution()
def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    return teams[-1][1]



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

    racers[0], racers[-1] = racers[-1], racers[0]



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

    no_of_attendees = len(arrivals)

    

    # Since indexing in python starts from zero, you have to  increase the index of the given name by one to 

    # match it with the value gotten from len(). Also (no_of_attendees/2) is rounded up to the nearest whole numbber

    # in the case of odd number divisions, so that we make sure the given a name is after at least half, in all scenarios.

    

    if arrivals.index(name)+1 > (round(no_of_attendees/2)) and (arrivals.index(name)+1) < no_of_attendees:

        return True

    else: 

        return False



q5.check()
#q5.hint()

q5.solution()