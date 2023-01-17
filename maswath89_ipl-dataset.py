# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
matches = pd.read_csv('/kaggle/input/ipl/matches.csv')

deliveries = pd.read_csv('/kaggle/input/ipl/deliveries.csv')
matches.head()
matches['team1'].unique()
matches['team1'] =matches['team1'].replace('Rising Pune Supergiant','Rising Pune Supergiants')

matches['team2'] =matches['team2'].replace('Rising Pune Supergiant','Rising Pune Supergiants')

matches['winner'] =matches['winner'].replace('Rising Pune Supergiant','Rising Pune Supergiants')

matches['toss_winner'] =matches['toss_winner'].replace('Rising Pune Supergiant','Rising Pune Supergiants')
matches.info()
import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(matches['season'], palette = 'rocket')
plt.figure(figsize= (12,6))

desc_order = matches['winner'].value_counts().sort_values(ascending = False).index

sns.countplot(matches['winner'], palette = 'rocket', order = desc_order)

plt.xticks(rotation = 'vertical')
plt.figure(figsize= (12,6))

desc_order = matches['team1'].value_counts().sort_values(ascending = False).index

sns.countplot(matches['team1'], palette = 'rocket', order = desc_order)

plt.xticks(rotation = 'vertical')
plt.figure(figsize= (30,10))

sns.countplot(matches['player_of_match'], palette = 'rocket')

plt.xticks(rotation = 'vertical')
matches['player_of_match'].value_counts()
matches['season'].value_counts()
matches[matches['win_by_runs'] == matches['win_by_runs'].max()]
desc = matches['venue'].value_counts().index

plt.figure(figsize= (12,6))

sns.countplot(matches['venue'], palette = 'rocket', order = desc)

plt.xticks(rotation = 'vertical')
new = matches[['id','team1','team2','winner']][(matches['team1']=='Chennai Super Kings') &(matches['team2'] == 'Mumbai Indians')]

#Pivot table to check the distribution of data

table = pd.pivot_table(new, values='id', index=['team1'], columns=['winner'],

                       aggfunc='count',fill_value=0,margins=False)

table
plt.figure(figsize = (12,6))

sns.countplot(matches['season'],hue = matches['toss_decision'])

plt.title('Toss Decision every season')

plt.xlabel('Season')

plt.ylabel('Decision Count')

#plt.xticks(rotation = 'horizontal')
team_list = matches['team2'].value_counts().index

team_list
total_matches = matches.groupby('team1')['id'].count().add(matches.groupby('team2')['id'].count())

matches_won   = matches.groupby('winner')['id'].count()

percentage_win = 100*(matches_won/total_matches)
concat = pd.concat([total_matches, matches_won, percentage_win], axis = 1).reset_index()

concat.columns = ['Team','Total Matches','Matches Won','Percentage Win']

concat
matches['toss_win_winner'] = 'No'

matches['toss_win_winner'].loc[matches['toss_winner'] == matches['winner']] = 'Yes'

toss_win_winner = matches['toss_win_winner'].value_counts()

labels = (np.array(toss_win_winner.index))

pct = (np.array((toss_win_winner / toss_win_winner.sum())*100))

plt.pie(pct, labels = labels)
plt.figure(figsize = (12,6))

sns.countplot('toss_winner',hue = 'toss_win_winner', data = matches)

plt.xticks(rotation = 'vertical')
matches['result'].value_counts()
deliveries.head()
deliveries[['match_id']][deliveries['match_id'] ==1]