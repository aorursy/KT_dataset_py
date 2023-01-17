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

    pass



# Check your answer

q1.check()
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    if L[1] :

        return L[1]

    else :

        return None

    pass



a = select_second([1 ,2 ,3])

print(a)
q1.hint()

q1.solution()
def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    return teams[-1][1]

    pass



# Check your answer

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

    #recers = racers[::-1]

    racers[0] , racers[-1] = racers[-1] , racers[0]

    

    pass



# Check your answer

q3.check()
racers =  ['L', 'M'] 

print(racers[::-1])
#q3.hint()

#q3.solution()
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]



# Put your predictions in the list below. Lengths should contain 4 numbers, the

# first being the length of a, the second being the length of b and so on.

lengths = [3 , 2 , 0 , 2]



# Check your answer

q4.check()
def fashionably_late(arrivals, name):

    """Given an ordered list of arrivals to the party and a name, return whether the guest with that

    name was fashionably late.

    """

    return arrivals.index(name) >= len(arrivals) / 2 and arrivals.index(name) != len(arrivals) - 1

        

    pass



# Check your answer

q5.check()
# line below provides some explanation

#q4.solution()
q5.hint()

q5.solution()