import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
import io

train=pd.read_csv('../input/ipl/matches.csv')

train.info()
train.dtypes
train.isnull().sum()
# checking whether there is any null value in the winner col

train[pd.isnull(train['winner'])]
# Filling the null values as Tie

train['winner'].fillna('Draw',inplace=True)
# checking the null values in the city col

train[pd.isnull(train['city'])]
# Filling the null cities as Dubai

train['city'].fillna('Dubai',inplace=True)

train.describe()
# Let us get some basic stats #

print("Number of matches played so far : ", train.shape[0])

print("Number of seasons : ", len(train.season.unique()))

sns.countplot(x='season', data=train)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='venue', data=train)

plt.xticks(rotation='vertical')

plt.show()


temp = pd.melt(train, id_vars=['id','season'], value_vars=['team1', 'team2'])



plt.figure(figsize=(12,6))

sns.countplot(x='value', data=temp)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='winner', data=train)

plt.xticks(rotation='vertical')

plt.show()
champ = train.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)

champ
toss = train.toss_decision.value_counts()

labels = (np.array(toss.index))

sizes = (np.array((toss / toss.sum())*100))

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss decision percentage")

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='toss_decision', data=train)

plt.xticks(rotation='vertical')

plt.show()
num_of_wins = (train.win_by_wickets>0).sum()

num_of_loss = (train.win_by_wickets==0).sum()

labels = ["Wins", "Loss"]

total = float(num_of_wins + num_of_loss)

sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Win percentage batting second")

plt.show()
train["field_win"] = "win"

train["field_win"].loc[train['win_by_wickets']==0] = "loss"

plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='field_win', data=train)

plt.xticks(rotation='vertical')

plt.show()
mom = train.player_of_match.value_counts()[:10]

labels = np.array(mom.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(mom), width=width, color='y')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Most number of M.O.M")

plt.show()
ump = pd.melt(train, id_vars=['id'], value_vars=['umpire1', 'umpire2'])



ump = ump.value.value_counts()[:10]

labels = np.array(ump.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(ump), width=width, color='r')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Umpires")

plt.show()
train['toss_winner_is_winner'] = 'no'

train['toss_winner_is_winner'].loc[train.toss_winner == train.winner] = 'yes'

result = train.toss_winner_is_winner.value_counts()



labels = (np.array(result.index))

sizes = (np.array((result / result.sum())*100))

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss winner is match winner")

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='toss_winner', hue='toss_winner_is_winner', data=train)

plt.xticks(rotation='vertical')

plt.show()
scores = pd.read_csv('../input/ipl/deliveries.csv')
runs = scores.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

runs = runs.iloc[:10,:]



labels = np.array(runs['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(runs['batsman_runs']), width=width, color='blue')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top run scorers in IPL")

plt.show()
boundaries = scores.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

boundaries = boundaries.iloc[:10,:]



labels = np.array(boundaries['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(boundaries['batsman_runs']), width=width, color='green')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of boundaries.!")

plt.show()


sixes = scores.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

sixes = sixes.iloc[:10,:]



labels = np.array(sixes['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(sixes['batsman_runs']), width=width, color='m')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of sixes.!")

plt.show()
dots = scores.groupby('batsman')['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

dots = dots.iloc[:10,:]



labels = np.array(dots['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(dots['batsman_runs']), width=width, color='c')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of dot balls.!")

plt.show()
balls = scores.groupby('bowler')['ball'].agg('count').reset_index().sort_values(by='ball', ascending=False).reset_index(drop=True)

balls = balls.iloc[:10,:]



labels = np.array(balls['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(balls['ball']), width=width, color='cyan')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Bowlers - Number of balls bowled in IPL")

plt.show()
dots1 = scores.groupby('bowler')['total_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='total_runs', ascending=False).reset_index(drop=True)

dots1 = dots1.iloc[:10,:]



labels = np.array(dots1['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(dots1['total_runs']), width=width, color='yellow')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Bowlers - Number of dot balls bowled in IPL")

plt.show()
extras = scores.groupby('bowler')['extra_runs'].agg(lambda x: (x>0).sum()).reset_index().sort_values(by='extra_runs', ascending=False).reset_index(drop=True)

extras = extras.iloc[:10,:]



labels = np.array(extras['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(extras['extra_runs']), width=width, color='magenta')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Bowlers with more extras in IPL")

plt.show()

plt.figure(figsize=(12,6))

sns.countplot(x='dismissal_kind', data=scores)

plt.xticks(rotation='vertical')

plt.show()