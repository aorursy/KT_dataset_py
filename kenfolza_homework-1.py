# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#fetching data
cupsData=pd.read_csv("../input/WorldCups.csv")
matchesData=pd.read_csv("../input/WorldCupMatches.csv")

# getting data info
cupsData.info()
matchesData.info()
# 1.Bar Plot (World Cup Winners)
cupsData['Winner'].value_counts().plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number Of Wins')
plt.title('World Cup Winners')
plt.show()
# 2.Bar Plot (Success of Host Countries)

# In this section, I wanted to analyse the success of host countries.
# I've got Winner, Runners-Up, Third and Fourth country data for each year.
# I decided to give 1 point for winner, 0.8 point for Runners-Up, 0.5 point for Third and 0.2 point for Fourth
def CalculateSuccess(row):
    if row['Country'] == row['Winner']:
        return 1.0
    elif row['Country'] == row['Runners-Up']:
        return 0.8 
    elif row['Country'] == row['Third']:
        return 0.5
    elif row['Country'] == row['Fourth']:
        return 0.2
    else:
        return 0.0
    
# Adding new column named 'HostCountrySuccess' and filling data by CalculateSuccess function
cupsData['HostCountrySuccess'] = cupsData.apply(CalculateSuccess, axis=1)

# Plotting the graph which shows the count of hosting the cup for each country with their success points
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.4
cupsData.groupby([cupsData.Country])['HostCountrySuccess'].sum().sort_index().plot(kind='bar', color='green', ax=ax, width=width, position=0,alpha=0.4)
cupsData['Country'].value_counts().sort_index().plot(kind='bar', color='blue', ax=ax, width=width, position=1,alpha=0.7)
ax.set_ylabel('Hosting Count / Succes Points')
plt.title('Host Country Success Graph')
plt.show()

# Blue bar shows the count of hostiong the cup, and the green bar shows success ratio.
# 3.Heat Map (Correlation)

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(cupsData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.title('World Cup Data Heat Map')
plt.show()

# 4. Scatter Plot of Goals
cupsData.plot(kind='scatter',x='MatchesPlayed',y='GoalsScored',alpha='0.5',color='red')
plt.xlabel('Matches Played')
plt.ylabel('Goals Scored')
plt.title('Scatter Plot of Goals')
plt.show()

# 5. Histogram of Goals Scored

# Histogram graph of total goals scored in each game during the world cup history.
# Creating a new column named 'TotalGoalsInMatch' by adding 'Home Team Goals' with 'Away Team Goals'
matchesData['TotalGoalsInMatch']=matchesData['Home Team Goals'] + matchesData['Away Team Goals']
matchesData.TotalGoalsInMatch.plot(kind = 'hist',bins = 50)
plt.title('Histogram of Goals Scored')
plt.xlabel('Goals Scored')
plt.show()
