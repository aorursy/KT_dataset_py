# Number Guessing Game - part 1 (computer picks the number) 



import random



#pick generate random number between 1 to 9

rannum = random.randint(1,9)

print("Random Number : ",rannum)

playing = True

count = 5



# loop while not correct

while(playing):

    # input users guess

    action = int(input())

    

    if count > 0:

        # tell user higher or lower

        if (action > rannum):

            print("User number : Higher")

        elif (action < rannum):

            print("User number : Lower")

        elif (action == rannum):

            count = 0

            print("You WIN!!!!")

            playing = False

    elif count <=0:

        print("You used 5 changes; Game OVER!!!")

        playing = False

    count = count - 1

    

else:

    print("Game completed!!!!")



    



# tell user higher or lower

# extra credit track the users guesses and the lose if the go over a limit of 5

# chances = 0

# show the user the won if they are correct and end the game

# extra credit:

# allow them to play again

# keep history of how many guesses they made for each time the played and whether they won or lost.
# Part 2 - Computer guesses your number

# Computer guess your number

# human enter enter 'h' for higher 'l' for lower 'c' for correct 

# if nots correct computer can guess for 10 times

# print list what computer guessed and most frequent number computer guessed



import random

import bisect

import statistics

from statistics import mode



def most_common(list):

    return(mode(list))



print("Pick number from 1 to 9: ",end = " ")

picknumber = int(input())

count = 10

comguess = []

l=[]

while (count > 0 ):    

    guessnum = random.randint(1,9)

    comguess.append(guessnum)

    #Sorting the computer guess number and inserting to list l

    bisect.insort(l,guessnum)

    print("Computer Guess : ", guessnum)

    print("enter 'h' for higher 'l' for lower 'c' for correct : ",end=" ")

    result=input()

    if result == 'c':

        print("Computer guessed correct!!!!")

        break

    else:

        count = count-1

    

print("Game Over!!!!")

print("Computer guess List : ",comguess)

print("Sorted List :", l)

print("Most common pick :", most_common(l))