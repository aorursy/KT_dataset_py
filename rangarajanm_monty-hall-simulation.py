# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random
def binarize(num):

	#binarize function for the three doors

	if num == 1:

		return [0,0,1]

	elif num == 2:

		return [0,1,0]

	else:

		return [1,0,0]
def rand_outcome():

	# Randomize the door in which the prize is present

	#random.seed(random.randint(1,1e12))

  	outcome = random.sample([1,2,4],1)[0]

	#print ("Outcome = %d"%(outcome))

  	return binarize(outcome)
def user_selection():

	# Randomize the door that the user picks the first time

	#seed = random.randint(1,1e12)

	sel = random.sample([0,1,2],1)[0]

	#print ("User selection = %d"%(sel))

	return sel
def MontyHallNoSwitch():

	# This returns the outcome (win or loss for the user under No-switch strategy)

	user_pick_int = user_selection()

	outcome = rand_outcome()

	return outcome[user_pick_int]
def MontyHallSwitch():

	# This returns the outcome (win or loss for the user under Switch strategy)

	user_pick_int = user_selection()

	outcome = rand_outcome()

	# The host is going to open one of the doors other than what the guest picked

	choices = [0,1,2]

	choices.pop(user_pick_int)

	# print (choices)

	# Since the host always opens the door without the prize, the guest loses

	# the game by switching if their original choice was the correct door

	if (outcome[choices[0]] == 0 and outcome[choices[1]] == 0):

		return 0

	else:

		return 1
def MontyHallSimulation(iter=10000):

	winsNoSwitch, winsSwitch = 0.0 , 0.0

	for i in range(0,iter):

		winsNoSwitch = winsNoSwitch + MontyHallNoSwitch()

		winsSwitch = winsSwitch + MontyHallSwitch()

		if i == 0:

			print ("Probability of winning after N iterations  NoSwitch Stategy  Switch Strategy")

		elif i%100000 == 0:

			print ("Prob. of winning after {0} iterations  {1}  {2}".format(i,winsNoSwitch/i, winsSwitch/i))

	return (winsNoSwitch/iter, winsSwitch/iter)
winProbNoSwitch, winProbSwitch = MontyHallSimulation(1000000)

print ("Probability of winning without switching is {}".format(float(winProbNoSwitch)))

print ("Probability of winning with switching is {}".format(float(winProbSwitch)))
def MontyHallSimulationData(iter=10000):

    winsNoSwitch, winsSwitch = 0.0 , 0.0

    switchWinsProb = []

    noswitchWinsProb = []

    for i in range(0,iter):

        winsNoSwitch = winsNoSwitch + MontyHallNoSwitch()

        noswitchWinsProb.append(winsNoSwitch/(i+1))

        winsSwitch = winsSwitch + MontyHallSwitch()

        switchWinsProb.append(winsSwitch/(i+1))

    return (noswitchWinsProb, switchWinsProb)
noswitchWinsProb, switchWinsProb = MontyHallSimulationData(10000)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df1 = pd.DataFrame({'iter':range(10000),'winProb':noswitchWinsProb})

df2 = pd.DataFrame({'iter':range(10000),'winProb':switchWinsProb})
df1.tail()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.lineplot(x='iter',y='winProb',data=df1)

plt.subplot(1,2,2)

sns.lineplot(x='iter',y='winProb',data=df2)