# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read the input files and look at the top few lines #

data_path = "../input/"

match_df = pd.read_csv(data_path+"matches.csv")

score_df = pd.read_csv(data_path+"deliveries.csv")



#Look at the top 5 rows of Match Data

match_df.head()
#Check the Score Data

score_df.head()
# Let us get some basic stats #

print("Number of matches played so far : ", match_df.shape[0])

print("Number of seasons : ", len(match_df.season.unique()))
score_df.columns

#Looks like only 1 column is hidden, let's take of the limit and see the data again
#set options for pandas to display max number of colums

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
score_df.head()
match_df.columns # Look at the column season, this should tell us about season count
match_df['season'].unique()# apply unique to the series to get a numpy array of unique values 

#if you would like to see a count get the len of the unique numpy array

len(match_df['season'].unique()) # totally there are 10 seasons data
seasons_years = match_df['season'].unique()

seasons_years.min()# 2008 is the first season
match_df['season'].value_counts()


sns.countplot(x='season', data=match_df)

plt.show()
match_df.venue.value_counts()
#Let's plot the data to see how it looks like

plt.figure(figsize=(12,8))

sns.countplot(x='venue', data=match_df,order = match_df.venue.value_counts().index)

plt.xticks(rotation='vertical')

plt.show()
len(match_df.venue.unique())
#Number of matches played by each team

temp_df = pd.melt(match_df, id_vars=['id','season'], value_vars=['team1', 'team2'])



print(temp_df.value.value_counts())

plt.figure(figsize=(10,6))

sns.countplot(x='value', data=temp_df,order = temp_df.value.value_counts().index)

plt.xticks(rotation='vertical')

plt.show()
#Number of wins per team

match_df.winner.value_counts()
#winner = match_df.winner.value_counts()

plt.figure(figsize=(12,6))

sns.countplot(x='winner', data=match_df,order = match_df.winner.value_counts().index)

plt.xticks(rotation='vertical')

plt.show()
# Toss Decision

match_df.toss_decision.value_counts()
toss = match_df.toss_decision.value_counts()

labels = (np.array(toss.index))

sizes = (np.array((toss / toss.sum())*100))

colors = ['green', 'grey']

plt.figure(figsize=(8,8))

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

plt.figure(figsize=(8,8))

colors = ['green', 'red']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Win percentage batting second")



plt.show()
match_df["field_win"] = "win"

match_df["field_win"].loc[match_df['win_by_wickets']==0] = "loss"

plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='field_win', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
manOmatch = match_df.player_of_match.value_counts()

manOmatch[:10]
score_df["bowl"] = 1

df_teams = score_df.groupby(["batting_team"]).sum().reset_index()

df_teams = df_teams[["batting_team", "wide_runs", "bye_runs", "legbye_runs", "noball_runs", "total_runs", "batsman_runs", "bowl"]]

plot_columns = ["batting_team", "total_runs", "batsman_runs", "bowl"]

df_runs = df_teams[plot_columns].sort_values(by=["total_runs"], ascending=[False])



f, ax = plt.subplots(figsize=(10, 7))

df_runs.plot.barh(ax=ax)

f.suptitle("Team's Runs made vs runs given", fontsize=24)

wrap = ax.set_yticklabels(list(df_runs["batting_team"]))
plot_columns = ["batting_team", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"]

df_extras = df_teams[plot_columns].sort_values(by=["wide_runs"], ascending=[False])



f, ax = plt.subplots(figsize=(10, 9))

df_extras.plot.barh(ax=ax)

f.suptitle("Extra Runs per Team", fontsize=24)

wrap = ax.set_yticklabels(list(df_extras["batting_team"]))

match_df['toss_winner_is_winner'] = 'no'

match_df['toss_winner_is_winner'].loc[match_df.toss_winner == match_df.winner] = 'yes'

temp_series = match_df.toss_winner_is_winner.value_counts()



labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

colors = ['green', 'cyan']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss winner is match winner")

plt.show()
plt.figure(figsize=(12,10))

sns.countplot(x='toss_winner', hue='toss_winner_is_winner', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
# create a function for labeling #

#Copied from SRK

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')
df_batsman = score_df.groupby(["batsman"]).sum().reset_index().nlargest(10, 'batsman_runs')



df_batsman = df_batsman[["batsman", "batsman_runs", "bowl"]]

df_batsman["Strike Rate"] = df_batsman.apply(

    lambda row: int((row["batsman_runs"] * 100) / row["bowl"]), axis=1)

df_batsman = df_batsman.sort_values(by=["batsman_runs"], ascending=[True])



f, ax = plt.subplots(figsize=(12, 6))

df_batsman[["batsman", "batsman_runs", "bowl"]].plot.barh(ax=ax)

f.suptitle("Top 10 Run Scorers", fontsize=24)

wrap = ax.set_yticklabels(list(df_batsman["batsman"]))



rects = ax.patches

bar_labels = list(df_batsman["Strike Rate"])



for i in range(len(bar_labels)):

  label = "Strike Rate: " + str(bar_labels[i])

  ax.text(1500, rects[i].get_y(), label, ha='center',

          va='bottom', size='smaller', color="white")
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

rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='r')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Batsman with most number of boundaries.!")

autolabel(rects)

plt.show()
df_dismissed = score_df.groupby(["dismissal_kind"]).sum(

).reset_index().nlargest(5, 'bowl').reset_index()



df_dismissed = df_dismissed[["dismissal_kind", "bowl"]]

df_dismissed = df_dismissed.rename(columns={'bowl': 'count'})

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

f.suptitle("Top 5 Dismissal Kind", fontsize=14)



df_dismissed.plot.bar(ax=ax1, legend=False)

ax1.set_xticklabels(list(df_dismissed["dismissal_kind"]), fontsize=8)



for tick in ax1.get_xticklabels():

  tick.set_rotation(0)



df_dismissed["count"].plot.pie(ax=ax2, labels=df_dismissed[

                               "dismissal_kind"], autopct='%1.1f%%', fontsize=8)

wrap = ax2.set_ylabel('')