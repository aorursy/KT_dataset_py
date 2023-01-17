# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex4 import *
print('Setup complete.')
def test(A):
    return 'PAR' if A % 2 == 0 else 'IMPAR' 

test(11)
L = [0,1,2,3,]
#x.__getitem__(y) <==> x[y]
L.__getitem__(2)
def select_second(L):
    """Return the second element of the given list. If the list has no second
    element, return None.
    """
    return None if len(L) <= 1 else L.__getitem__(1)
    pass

q1.check()
#q1.hint()
#q1.solution()
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]
hand_last = hands[-1]
print(hand_last[1])
def losing_team_captain(teams):
    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)
    from the last listed team
    """
    worst_team = teams[-1]
    return worst_team[1]
    pass

q2.check()
#q2.hint()
#q2.solution()
a = [1,2,3]
a[0], a[-1] = a[-1], a[0]
print(a)
def purple_shell(racers):
    """Given a list of racers, set the first place racer (at the front of the list) to last
    place and vice versa.    
    >>> r = ["Mario", "Bowser", "Luigi"]
    >>> purple_shell(r)
    >>> r
    ["Luigi", "Bowser", "Mario"]
    """
    racers[0], racers[-1] = racers[-1], racers[0]
    #return racers
    pass

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
arrivals=['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
print(len(arrivals) / 2,
      arrivals.index('May') + 1,
      arrivals.index(arrivals[-1]),
      sep='\n'
     )
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    total_guests = len(arrivals)
    arrival_order_guest = arrivals.index(name) + 1
    if total_guests % 2 == 0:
        return True if ( (arrival_order_guest > (total_guests / 2) ) and (arrival_order_guest != total_guests) ) else False
    else:
        return True if ( (arrival_order_guest > ( (total_guests + 1) / 2) ) and (arrival_order_guest != total_guests) ) else False
    pass

q5.check()
#q5.hint()
#q5.solution()
def count_negatives(nums):
    if(nums):
        return 1 + count_negatives(nums) if(nums.pop() < 0) else count_negatives(nums)
    else:
        return 0
count_negatives([11,2,3,2,4])

def count_negatives(nums):
    if (len(nums)==0):
        return 0
    else:
        return (1 if (nums[0]<0) else 0) + count_negatives(nums[1:])
count_negatives([0, -1, -1])
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    if (nums):
        #return 1 + count_negatives(nums) if(nums.pop() < 0) else count_negatives(nums)
        return aux + count_negatives(nums) if (nums.pop() < 0) else count_negatives(nums)       
    else:
        return 0
    ###  STOPPED HERE  ###
    pass

q6.check()
q6.hint()
q6.solution()