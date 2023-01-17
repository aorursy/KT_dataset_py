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
#Reading the dataset
dataset = pd.read_csv('../input/ipldata/matches.csv')
#Looking at the data
dataset.head()
#Deriving insights from the dataset
dataset.describe()
#Checking for null values
dataset.isnull().sum()
#Since more than 80% of the umpire3 column is missing, drop the column
dataset = dataset.drop('umpire3',axis=1)
#Check if the umpire3 column is dropped
dataset.head()
#Visualizing the city with the number of matches played
import matplotlib.pyplot as plt
import seaborn as sns
#Gives us a Series with the counts of each city in the dataset in the descending order of number of times they occur
cities = dataset['city'].value_counts()
#Changing it into a dataframe
cities = pd.DataFrame(cities)
#The indexes in this dataframe are the name of the cities. Changing the index into the first column with the name 'index' which contains the names of the cities and the second column 'city' where the number of matches played in each city is given
cities.reset_index(level=0,inplace=True)
plt.figure(figsize=(11,7))
c = sns.barplot(cities['index'],cities['city'])
c.set_xticklabels(labels=cities['index'],rotation=60)
#To get you an idea of what cities dataframe looks like
cities
winners = dataset['winner'].value_counts()
winners = pd.DataFrame(winners)
winners.reset_index(level=0,inplace=True)
winners = winners.rename(columns={'index': 'Cities','winner':'Matches Played'})
#Looking at what winners look like
winners
plt.figure(figsize=(11,7))
c = sns.barplot(winners['Cities'],winners['Matches Played'])
c.set_xticklabels(labels=winners['Cities'],rotation=60)
#Taking the total number of times each team has won 
wins = dataset['winner'].value_counts()
#Adding the columns team1 and team2 gives the total number of matches each team has played
team1 = dataset['team1'].value_counts() 
team2 = dataset['team2'].value_counts()
#Storing the total number of matches played by each team in a list
played = []
for i in team1.index:
    for j in team2.index:
        if i==j:
            played.append(team1[i] + team2[j])
#As the team1 column is what we checked the values with the order should be the same as that of team1
x = team1.index            
x = list(x)
#Creating a dataframe of the total number of matches played by each team
matches_played = pd.DataFrame(played,x)
new_wins = []
for i in matches_played.index:
    for j in wins.index:
        if(i==j):
            new_wins.append(wins[j])
new_wins = np.array(new_wins)
total = np.array(matches_played.values)
total = total.ravel()
#Calculating the win percentage of each team
percentage = new_wins/total
print(percentage)
win_percentage = pd.DataFrame(percentage,x)
win_percentage.reset_index(level=0,inplace=True)
win_percentage = win_percentage.rename(columns={'index':'Team',0:'Percentage'})
plt.figure(figsize=(11,7))
plt.ylim(0.2,0.7)
c = sns.barplot(win_percentage['Team'],win_percentage['Percentage'])
c.set_xticklabels(labels=win_percentage['Team'],rotation=60)