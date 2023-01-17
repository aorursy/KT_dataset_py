from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

    if len (L)>1:

        return L[1]

    else:

        return None

    

# Check your answer

q1.check()
#q1.hint()

#q1.solution()
l = [[1,2,3],[4,5,6]]

print (l[-1][1])

def losing_team_captain(teams):

    return teams[-1][1]



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
l = [1,2,3]

swp=l[0]

l[0] =l[-1]  

l[-1] = swp

print(swp)

print (l[0])





def purple_shell(racers):

    swp=racers[0]

    racers[0] =racers[-1]  

    racers[-1] = swp



  





# Check your answer

q3.check()
#q3.hint()

#q3.solution()
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]

lengths = [len (a) ,len (b),len (c),len (d)]





q4.check()
# line below provides some explanation

#q4.solution()
l = ['abd','ahmed','ali','salah','moath','osama']

x= l.index('moath')

print (x)
def fashionably_late(arrivals, name):

    gus = arrivals.index(name)

    half = len(arrivals)/2

    

    return (gus>=half and gus!=len(arrivals)-1)



# Check your answer

q5.check()
#q5.hint()

#q5.solution()