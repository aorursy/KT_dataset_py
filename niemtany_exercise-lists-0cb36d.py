# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex4 import *
print('Setup complete.')
def select_second(L):
    if len(L) < 2:
        return None
    return L[1]
q1.check()
#q1.hint()
#q1.solution()
def losing_team_captain(teams):
    return teams[-1][1]

q2.check()
q2.hint()
#q2.solution()
def purple_shell(racers):
    
    temp = racers[0]
    racers[0] = racers[-1]
    racers[-1] = temp

q3.check()
#q3.hint()
#q3.solution()
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# Put your predictions in the list below. Lengths should contain 4 numbers, the
# first being the length of a, the second being the length of b and so on.
lengths = [a]

q4.check()
# line below provides some explanation
#q4.solution()
def fashionably_late(arrivals, name):
    order = arrivals.index(name)
    return order >= len(arrivals) / 2 and order != len(arrivals) - 1
    pass

q5.check()
#q5.hint()
#q5.solution()
def count_negatives(nums):
    nums.append(0)
    nums = sorted(nums)
    return nums.index(0)

q6.check()
#q6.hint()
q6.solution()