from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    if(L[1] != None ):

        return L[1]

    else:

        return None





# Check your answer

list = [1,2,3]

# list = [1]



if(select_second(list) != None):

    print("the Second item of this List is ", select_second(list))

else:

    print("This List Contain One Item Only")
#q1.hint()

#Check there is actualy a second item in the list whether by Checking the second item if it's null or if the length of the list is greater than 1

#q1.solution()

def select_second(L):

    """Return the second element of the given list. If the list has no second

    element, return None.

    """

    if(L[1] != None ):

        return L[1]

    else:

        return None





# Check your answer

list = [1,2,3]

# list = [1]



if(select_second(list) != None):

    print("the Second item of this List is ", select_second(list))

else:

    print("This List Contain One Item Only")
def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    if(teams != None):

        return  teams[-1][1]

    else:

        print("There is No teams ")

    pass



# Check your answer

mylist = [

    ["Coach Mohammad ","Captain Ahmad","Player Firas"],

    ["Coach Rawan ","Captain Sara "," Player maryam"]

]



print(losing_team_captain(mylist))
#q2.hint()

# make a list of lists contain each team members 

#in the function check if there is an actual team , if there is return the second player name from the last team 

#q2.solution()

def losing_team_captain(teams):

    """Given a list of teams, where each team is a list of names, return the 2nd player (captain)

    from the last listed team

    """

    if(teams != None):

        return  teams[-1][1]

    else:

        print("There is No teams ")

    pass



# Check your answer

mylist = [

    ["Coach Mohammad ","Captain Ahmad","Player Firas"],

    ["Coach Rawan ","Captain Sara "," Player maryam"]

]



print(losing_team_captain(mylist))
def purple_shell(racers):

    """Given a list of racers, set the first place racer (at the front of the list) to last

    place and vice versa.



    # >>> r = ["Mario", "Bowser", "Luigi"]

    # >>> purple_shell(r)

    # >>> r

    # ["Luigi", "Bowser", "Mario"]

    """

    if(racers != None):

        racers[0],racers[-1] = racers[-1],racers[0]

    else:

        print("Racers List is Empty ")







# Check your answer

racers = ["Mario", "Bowser", "Luigi"]

purple_shell(racers)

print(racers)
#q3.hint()

#Check if the Racers list is empty , if not switch between the first item in the list with index 0 and last item which is index -1 else print the list is empty

#q3.solution()
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]



# Put your predictions in the list below. Lengths should contain 4 numbers, the

# first being the length of a, the second being the length of b and so on.

lengths = [3 ,2 , 0 ,2]



# Check your answer

print(len(a))

print(len(b))

print(len(c))

print(len(d))
# line below provides some explanation

# list a contains 3 items sperated by a comma , list b contain 2 containg a number and another list , list c is empty , list d contain 2 items which are only 2,3

#q4.solution()
def fashionably_late(arrivals, name):

    """Given an ordered list of arrivals to the party and a name, return whether the guest with that

    name was fashionably late.

    """

    if(arrivals != None):

        if(arrivals.__contains__(name)):

            if(arrivals.index(name) > (len(arrivals) / 2)):

                print("Guest ",name," Has Arrived Late !!")

            else:

                print("Guest Arrived on Time :D ")

        else:

            print("Sorry This Guest isint in the Arrivals List :( ")

    else:

        print("Guest List is Emprty ")



# Check your answer

party_attendees = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']

fashionably_late(party_attendees , 'Ford')
#q5.hint()

#check if the list of arrivals is empty if not , check if the guest are in the arrivals list , then check 

#if it's greater than the half of the half of the length of the list

#q5.solution()