# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import random



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# In this problem, there are three doors presented. Two doors reveal GOATS and one door reveal CAR.



monty_doors = ['car','goat','goat']

monty_doors
# shuffle to stimulate random trials

random.shuffle(monty_doors)

monty_doors
# Now,choose randomly between three doors. Assign 0,1,2 for first, second and third respectively.

door_index= random.choice([0,1,2])

door_index
# Now, get the results of staying with initial choice, and remove that option from the list of available doors.

stayer = monty_doors.pop(door_index)

stayer
# we can get other two doors that Monty has to choose from

monty_doors
# Now, Monty has to open the door that has a goat. 

# So, remove the first door with a goat behind it.

monty_doors.remove('goat')

monty_doors
# now only one remaining door left to open.

# Monty offer to switch from original results



switch = monty_doors[0]

switch
# Now try to do this 100000 times

n_trials = 100000

stayer_1 =[]

switcher_1=[]

for i in range(n_trials):

    monty_doors= ['car', 'goat','goat']

    random.shuffle(monty_doors)

    door_index=random.choice([0,1,2])

    stayer =monty_doors.pop(door_index)

    monty_doors.remove('goat')

    switcher = monty_doors[0]

    stayer_1.append(stayer)

    switcher_1.append(switcher)
# probability of winning chance for Stayer = 33%

prob_win_stay = stayer_1.count('car')/n_trials

prob_win_stay
# probability of winning chance for Switcher = 67%

prob_win_switch= switcher_1.count('car')/n_trials

prob_win_switch
Who_Wins = np.array([0.33, 0.67])

x_locations = [0,1]

box_labels = ['Stayer', 'Switcher']

plt.bar(x_locations, Who_Wins)

plt.xticks(x_locations, box_labels)

plt.ylabel("Probability of Winning")

plt.xlabel ("Who Wins")

plt.title('Monty Hall - Winning Probability Strategy')