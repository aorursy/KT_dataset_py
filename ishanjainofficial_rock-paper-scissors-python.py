# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from random import randint



#create a list of play options

t = ["Rock", "Paper", "Scissors"]



#assign a random play to the computer

computer = t[randint(0,2)]



#set player to False

player = False



while player == False:

#set player to True

    player = input("Rock, Paper, Scissors?")

    if player == computer:

        print("Tie!")

    elif player == "Rock":

        if computer == "Paper":

            print("You lose!", computer, "covers", player)

        else:

            print("You win!", player, "smashes", computer)

    elif player == "Paper":

        if computer == "Scissors":

            print("You lose!", computer, "cut", player)

        else:

            print("You win!", player, "covers", computer)

    elif player == "Scissors":

        if computer == "Rock":

            print("You lose...", computer, "smashes", player)

        else:

            print("You win!", player, "cut", computer)

    else:

        print("That's not a valid play. Check your spelling!")

    # player was set to True, but we want it to be False so the loop continues

    player = False

    computer = t[randint(0,2)]
#create a list of play options

t = ["Rock", "Paper", "Scissors"]



#assign a random play to the computer

computer = t[randint(0,2)]



#set player to False

player = False
while player == False:

#set player to True

    player = input("Rock, Paper, Scissors?")

    if player == computer:

        print("Tie!")

    elif player == "Rock":

        if computer == "Paper":

            print("You lose!", computer, "covers", player)

        else:

            print("You win!", player, "smashes", computer)

    elif player == "Paper":

        if computer == "Scissors":

            print("You lose!", computer, "cut", player)

        else:

            print("You win!", player, "covers", computer)

    elif player == "Scissors":

        if computer == "Rock":

            print("You lose...", computer, "smashes", player)

        else:

            print("You win!", player, "cut", computer)

    else:

        print("That's not a valid play. Check your spelling!")

    #player was set to True, but we want it to be False so the loop continues

    player = False

    computer = t[randint(0,2)]