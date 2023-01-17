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
    return L[1] if len(L) > 1 else None

q1.check()
#q1.hint()
#q1.solution()
def losing_team_captain(teams):
    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)
    from the last listed team
    """
    return teams[-1][1] # last team, 2nd item

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
    print("arrivals:", arrivals)
    print("length of arrivals:", len(arrivals))
    print("halfway point:", len(arrivals) / 2)
    print("last arrival:", arrivals[-1])
    print("name value:", name)
    print("arrivals index of name:", arrivals.index(name))
    print("value to return:", ( (arrivals.index(name) > (len(arrivals) / 2)) and name != arrivals[-1]))
    #print(( (arrivals.index(name) > (len(arrivals) / 2)) and name != arrivals[-1]))
    if ( (arrivals.index(name) >= (len(arrivals) / 2)) and name != arrivals[-1]):
        return True
    else:
        return False

q5.check()
#q5.hint()
q5.solution()
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    print("Length of nums:", len(nums))
    print("nums index of zero:", my_list.index(0))
    print("nums contains 0:", nums.__contains__(0))
    if nums.__contains__(0) == False:
        nums.append(0)
    print("nums:", nums)
    print("nums contains 0:", nums.__contains__(0))  
    nums.sort()
    print("nums:", nums)
    print("values less than zero:", nums.index(0))
    return nums.index(0)
q6.check()
q6.hint()
my_list = [-2, -1, 3, 5]
print("Length of my_list:", len(my_list))
#print("my_list index of zero:", my_list.index(0))
print("my_list contains 0:", my_list.__contains__(0))
if my_list.__contains__(0) == False:
    my_list.append(0)
print("my_list:", my_list)
print("my_list contains 0:", my_list.__contains__(0))  
my_list.sort()
print("my_list:", my_list)
print("my_list index of zero:", my_list.index(0))
#return my_list.index(0)
#help(list)
#q6.solution()