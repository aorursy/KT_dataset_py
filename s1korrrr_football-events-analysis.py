# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


#Importing the libraries



import matplotlib.pyplot as plt # basic plotting

%matplotlib inline

import seaborn as sns # more plotting

sns.set(style="whitegrid")



#Importing the data

events = pd.read_csv("../input/events.csv")

games = pd.read_csv("../input/ginf.csv")
events.head()
goals=events[events["is_goal"]==1]
fig=plt.figure(figsize=(8,6))

plt.hist(goals.time,width=1,bins=100,color="green")   #100 so 1 bar per minute

plt.xlabel("Minutes")

plt.ylabel("Number of goals")

plt.title("Number of goals against Time during match")
goals_winning_team = []



# go row by row (don't think a one liner would work)

for I,game in games.iterrows():

    if game['fthg'] > game['ftag']:

        goals_winning_team.append(game['fthg'])

    elif game['fthg'] < game['ftag']:

        goals_winning_team.append(game['ftag'])

        

avg_goals = np.mean(goals_winning_team)

std_goals = np.std(goals_winning_team)

        

print("Average Goals Per Game from Winning Team: %0.3f" % avg_goals)

print("Std. of Goals per game from winning team: %0.3f" % std_goals)

print("%% Deviation: +/- %0.3f%%" % ((std_goals/avg_goals)*100))
plt.hist(goals_winning_team, bins=10, normed=True)

plt.show()