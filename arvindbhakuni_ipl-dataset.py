import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
players = pd.read_excel('../input/ipl-data-set/Players.xlsx')

matches = pd.read_csv('../input/ipl-data-set/matches.csv')
players.head(2)
players.info()
matches.drop("umpire3", axis=1, inplace=True)
matches.dropna(inplace=True)
players.dropna(inplace=True)
matches.set_index("id", drop=True, inplace=True)
matches.Season = matches.Season.str.replace('IPL-', '').astype('int')
players.Batting_Hand = players.Batting_Hand.str.replace("_Hand", "")
matches.date
matches['date']= pd.to_datetime(matches['date'])
players.head()
players_per_country = players.Country.value_counts()

players_per_country
plt.figure(figsize=(10,5))

sns.set_style("whitegrid")

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})



sns.countplot(x='Country', data=players, palette='dark')

plt.xticks(rotation=75)

plt.title('Countries vs Players')

plt.grid()

plt.show()
plt.figure(figsize=(10,5))

sns.set_style("whitegrid")

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})



sns.countplot(x='Country', data=players, hue='Batting_Hand', palette='coolwarm')

plt.xticks(rotation=75)

plt.grid()

plt.show()
players.Batting_Hand.value_counts()
plt.figure(figsize=(10,5))

sns.set_style("whitegrid")



sns.countplot(x='Batting_Hand', data=players, palette='cubehelix')

plt.title('Right vs Left hand Batsmen')
players.head()
plt.figure(figsize=(10,6))

players.DOB.hist(bins=60, color='purple')

plt.title('Players born per year')
sns.catplot(x='Bowling_Skill', y='DOB', hue='Batting_Hand', data=players, kind='swarm', aspect=2)

plt.xticks(rotation=75)

plt.title('Bowling skills vs Player born')

plt.show()
sns.catplot(x='Batting_Hand', y='DOB', hue='Country', data=players, kind='swarm', aspect=2)

plt.xticks(rotation=75)

plt.title('Batting Skills vs Players born')

plt.show()
matches.head(3)
groupby_year = matches.groupby('Season')['winner'].value_counts()

groupby_year
plt.figure(figsize=(10,6))

groupby_year.loc[2019].plot(kind='bar')

plt.xlabel(None)

plt.ylabel('Number of Matches')

plt.title('Wins by Teams in year 2008', size=18)

plt.tight_layout()
matches.winner.value_counts()
plt.figure(figsize=(10,6))

matches.winner.value_counts().plot(kind='bar', color='orange')

plt.xlabel(None)

plt.ylabel('Number of Matches')

plt.title('Wins by Teams in all Seasons', size=18)

plt.tight_layout()