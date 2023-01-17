# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# create a function for labeling #

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')
# read the input files and look at the top few lines #

data_path = "../input/"

match_df = pd.read_csv(data_path+"matches.csv")

score_df = pd.read_csv(data_path+"deliveries.csv")

match_df.head()
# Let us get some basic stats #

print("Number of matches played so far : ", match_df.shape[0])

print("Number of seasons : ", len(match_df.season.unique()))
sns.countplot(x='season', data=match_df)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='venue', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
temp_df = pd.melt(match_df, id_vars=['id','season'], value_vars=['team1', 'team2'])



plt.figure(figsize=(12,6))

sns.countplot(x='value', data=temp_df)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='winner', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
temp_df = match_df.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)

temp_df
temp_series = match_df.toss_decision.value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss decision percentage")

plt.show()

          
plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='toss_decision', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
num_of_wins = (match_df.win_by_wickets>0).sum()

num_of_loss = (match_df.win_by_wickets==0).sum()

labels = ["Wins", "Loss"]

total = float(num_of_wins + num_of_loss)

sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Win percentage batting second")

plt.show()
match_df["field_win"] = "win"

match_df["field_win"].ix[match_df['win_by_wickets']==0] = "loss"

plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='field_win', data=match_df)

plt.xticks(rotation='vertical')

plt.show()


temp_series = match_df.player_of_match.value_counts()[:10]

labels = np.array(temp_series.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_series), width=width, color='y')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top player of the match awardees")

autolabel(rects)

plt.show()
temp_df = pd.melt(match_df, id_vars=['id'], value_vars=['umpire1', 'umpire2'])



temp_series = temp_df.value.value_counts()[:10]

labels = np.array(temp_series.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_series), width=width, color='r')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Umpires")

autolabel(rects)

plt.show()
match_df['toss_winner_is_winner'] = 'no'

match_df['toss_winner_is_winner'].ix[match_df.toss_winner == match_df.winner] = 'yes'

temp_series = match_df.toss_winner_is_winner.value_counts()



labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss winner is match winner")

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='toss_winner', hue='toss_winner_is_winner', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
score_df.head()
temp_df = score_df.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='blue')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top run scorers in IPL")

autolabel(rects)

plt.show()
temp_df = score_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='green')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of boundaries.!")

autolabel(rects)

plt.show()
temp_df = score_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='m')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of sixes.!")

autolabel(rects)

plt.show()
temp_df = score_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['batsman'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='c')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of dot balls.!")

autolabel(rects)

plt.show()
def balls_faced(x):

    return len(x)



def dot_balls(x):

    return (x==0).sum()



temp_df = score_df.groupby('batsman')['batsman_runs'].agg([balls_faced, dot_balls]).reset_index()

temp_df = temp_df.ix[temp_df.balls_faced>200,:]

temp_df['percentage_of_dot_balls'] = (temp_df['dot_balls'] / temp_df['balls_faced'])*100.

temp_df = temp_df.sort_values(by='percentage_of_dot_balls', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

labels = np.array(temp_df['batsman'])

ind = np.arange(len(labels))

width = 0.9

rects = ax1.bar(ind, np.array(temp_df['dot_balls']), width=width, color='brown')

ax1.set_xticks(ind+((width)/2.))

ax1.set_xticklabels(labels, rotation='vertical')

ax1.set_ylabel("Count of dot balls", color='brown')

ax1.set_title("Batsman with highest percentage of dot balls (balls faced > 200)")

ax2.plot(ind+0.45, np.array(temp_df['percentage_of_dot_balls']), color='b', marker='o')

ax2.set_ylabel("Percentage of dot balls", color='b')

ax2.set_ylim([0,100])

ax2.grid(b=False)

plt.show()
temp_df = score_df.groupby('bowler')['ball'].agg('count').reset_index().sort_values(by='ball', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['ball']), width=width, color='cyan')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Bowlers - Number of balls bowled in IPL")

autolabel(rects)

plt.show()
temp_df = score_df.groupby('bowler')['total_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='total_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['total_runs']), width=width, color='yellow')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Bowlers - Number of dot balls bowled in IPL")

autolabel(rects)

plt.show()
temp_df = score_df.groupby('bowler')['extra_runs'].agg(lambda x: (x>0).sum()).reset_index().sort_values(by='extra_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['extra_runs']), width=width, color='magenta')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Bowlers with more extras in IPL")

autolabel(rects)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='dismissal_kind', data=score_df)

plt.xticks(rotation='vertical')

plt.show()