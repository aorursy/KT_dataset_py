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
    if len(L) >= 2:
       return L[1]
    else:
       return None
q1.check()
l = [1,2]
select_second(l)
#q1.hint()
#q1.solution()
sport_team_1 = ["CCoach","Captain1",'Player 1','Player 2', 'Rank1' ]
sport_team_2 = ["ACoach","Captain2",'Player 1','Player 2' , 'Rank2' ]
sport_team_3 = ["BCoach","Captain3",'Player 1','Player 2' , 'Rank3' ]
sport = [sport_team_1,sport_team_2,sport_team_3 ]
sport[-1][1]
def losing_team_captain(teams):
    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)
    from the last listed team
    """
    return teams[-1][1]

q2.check()
teams=[['Paul', 'John', 'Ringo', 'George']]
losing_team_captain(teams)
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
    first = racers[0]
    last = racers[-1]
    racers[-1] = first
    racers[0] = last
    racers

q3.check()
racers = ["Mario", "Bowser", "Luigi"]
#purple_shell(r)

first = racers[0]
print(first)
last = racers[-1]
print(last)
racers[-1] = first
racers[0] = last
racers

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
    ln = len(arrivals)//2 + 1 
    ix = arrivals.index(name)
    
    if ix >= ln:
       print("fashionably late")
       return True
    else:
       print("Not")
       return False

q5.check()
party_attendees = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
ln = len(party_attendees)//2 + 1 
#print(ln)

ix = party_attendees.index('Ford')
if ix >= ln:
   print("fashionably late")
else:
   print("Not") 

def fashionably_late(arrivals, name):
    order = arrivals.index(name)
    return order >= len(arrivals) / 2 and order != len(arrivals) - 1
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
    z_num = nums.index(0)
    return len(nums[0:z_num])

q6.check()
q6.hint()
#q6.solution()