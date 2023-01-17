from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

        if len(L)<2:

            return "none"

        return L[1]

    



L=[1,2]

print(select_second(L))
def select_second(L):

        return L[1]

    



L=[1,2,3]

print(select_second(L))
#q1.hint()

#q1.solution()
def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    i=teams[len(teams)-1]

    return i[1]



teams = [['J', 'M', 'Q'], ['R', 'T', 'S'], ['U', 'N', 'K']]

print(losing_team_captain(teams))
#q2.hint()

#q2.solution()
def purple_shell(racers):

    racers[0],racers[2]=racers[2],racers[0]



r = ["Mario", "Bowser", "Luigi"]

purple_shell(r)

print(r)
#q3.hint()

#q3.solution()
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]



# Put your predictions in the list below. Lengths should contain 4 numbers, the

# first being the length of a, the second being the length of b and so on.

lengths = [3 ,2 , 0 ,2]



print(len(a))

print(len(b))

print(len(c))

print(len(d))
# line below provides some explanation

#q4.solution()
def fashionably_late(arrivals, name):

    L=arrivals[int(len(arrivals)/2)+1:-1]

    if name in L:

        return "fashionably late"

    return "not fashionably late"



party_attendees = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']

print(fashionably_late(party_attendees, "Gilbert"))

print(fashionably_late(party_attendees, "Mona"))

print(fashionably_late(party_attendees, "Ford"))

print(fashionably_late(party_attendees, "May"))

print(fashionably_late(party_attendees, "Fleda"))
#q5.hint()

#q5.solution()