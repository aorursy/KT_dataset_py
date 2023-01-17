# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

from time import time as TT

from random import choice as ch

import numpy as np

import warnings

%matplotlib inline

ac = []

tc = []

N = []

st = TT()







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
if simulations == int(input("enter no of simulations you want to run-  ")) != int:

    simulations = int(3)



if doors == int(input("enter no of doors you want to have-  ")) != int:

    doors = 3



cards = list(range(1,doors))

print(cards)

for M in range(1,simulations): #Outer loop from 1 to no of simulations

    st1 = TT()

    score = []

    runs = 0

    with warnings.catch_warnings():

        warnings.filterwarnings("ignore", "Mean of empty slice")

    for K in range(1,M): # sub loop that simulates 1 to M(outerloop) games

        aset = []

        host = cards.copy()

        hbk = ch(host) #Randomly choose as answer which host knows

        aset.append(hbk)

        print("The host knows the car is behind door no",hbk)

        player = cards.copy()

        px = ch(player) # Contestants random guess

        aset.append(px)

        print ("Player's first choice is door no",px)

        chance = 0

        for i in host: # The computation....host will eliminate P(X|DOOR) = 0

            if i not in aset:

                chance = i

        print ("Host eliminates door no",chance)

        #print (player)

        player.pop(player.index(chance))

        player.pop(player.index(px))

        

        print ("current options",player)

        py= ch(player)

        print("player chooses door no",py)

        if py == hbk:

            score.append(1)

        else:

            score.append(0)

        runs = K

        print ("\n\n")

    ac.append(np.mean(score))

    N.append(M)

    en1 = TT()

    tc.append(en1-st1)

 
en = TT()   
print ("Total time for Loop  ", en - st )
plt.plot(N,ac)

plt.show()



plt.plot(N,tc)

plt.show()



print ("Averge Wins",np.nanmean(ac))