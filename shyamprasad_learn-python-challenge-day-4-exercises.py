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
    return None if len(L) < 2 else L[1]
q1.check()
#q1.hint()
#q1.solution()
def losing_team_captain(teams):
    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)
    from the last listed team
    """
    Last_Team = teams[-1]
    return Last_Team[1]
    #return teams[-1][1]

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
    racers[-1], racers[0] = racers[0], racers[-1]

q3.check()
#q3.hint()
#q3.solution()
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# Put your predictions in the list below. Lengths should contain 4 numbers, the
# first being the length of a, the second being the length of b and so on.
# predicted values lengths = [3, 2, 0, 2]
lengths = [len(a), len(b), len(c), len(d)]

q4.check()
# line below provides some explanation
#q4.solution()
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    Half_Idx = len(arrivals) / 2
    Last_Idx = len(arrivals)-1   
    if(name in arrivals):
        Name_Idx = arrivals.index(name)        
        if(Name_Idx >= Half_Idx and Name_Idx != Last_Idx):
            result = True
        else:
            result = False        
        print ('\n\nArrivals: ', arrivals, '\nName picked: ', name, '\nHalf_Idx: ', Half_Idx, 'Name_Idx: ', Name_Idx, 'Last_Idx: ', Last_Idx, '\nResult: ', result )
        return result
fashionably_late(['Paul', 'John', 'Ringo'], 'John')
#q5.check()
#q5.hint()
q5.solution()
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    
    try:
        if len(negative_list) == 0:
            print(negative_list)
            #negative_list = []
    except UnboundLocalError:
        negative_list = []
        print('ddd', len(negative_list))
       

    if len(nums) == 0:        
        return len(negative_list)
    else:        
        nums = sorted(nums)
        if nums[0] < 0:            
            negative_list.append(nums[0])                 
            nums.remove(nums[0])     
        else:            
            nums.clear()            
        print(negative_list, len(negative_list))
        count_negatives(nums)

print(count_negatives([5, -1, -2, 0, 3]))

#q6.check()
#q6.hint()
q6.solution()