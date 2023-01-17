# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/matches.csv')
df.head()
df.shape
#unique values for result
print(df.result.unique())
#List of teams played in IPL
print(df.team1.unique())
#Match results
print("No. of Tie matches: "+str(df[df.result=='tie'].id.count()))
print("No. of no result matches: "+str(df[df.result=='no result'].id.count()))
df.describe()
#matches per season
df.groupby('season')['season'].count()
#no of matches where toss winner is the match winner
print("No of matches where toss winner is match winner: "+str(df[(df.result == 'normal') & (df.toss_winner == df.winner)].id.count()))
print("No of matches where toss winner is not match winner: "+str(df[(df.result == 'normal') & (df.toss_winner != df.winner)].id.count()))
fig = plt.figure() 
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.3

df[(df.result == 'normal') & (df.toss_winner == df.winner)].groupby('venue')['venue'].count().plot(figsize=(20,10),kind='bar', color='blue', ax=ax, width=width, position=1,yticks=[0,5,10,15,20,25,30,35])
df[(df.result == 'normal') & (df.toss_winner != df.winner)].groupby('venue')['venue'].count().plot(figsize=(20,10),kind='bar', color='red', ax=ax2, width=width, position=0,yticks=[0,5,10,15,20,25,30,35])

plt.show()
df.groupby("winner")['winner'].count().plot(figsize=(12,12),kind='pie',autopct='%1.1f%%',shadow=True)
#Top 10 man of the match graph
df1 = pd.DataFrame({"count":df.groupby('player_of_match')['player_of_match'].count()}).reset_index()
df1 = df1.sort_values('count',ascending=False)
df1[0:10].plot.barh(figsize=(20,10),x='player_of_match',y='count',xticks=[2,4,6,8,10,12,14,16,18,20])

df_runs = pd.read_csv('../input/deliveries.csv')
df_runs.head()
df_runs.shape
df_bats = df_runs[df_runs.batsman == 'CH Gayle']
df_bats.head()
df1 = pd.DataFrame({"count":df_bats.groupby(['batting_team','match_id'])['batting_team'].count()}).reset_index()
df1.head()
df1.groupby('batting_team')['batting_team'].count()
df_bats.groupby('match_id')['batsman_runs'].sum().plot(figsize=(20,10),kind = 'line')
df_runs[df_runs.player_dismissed == 'CH Gayle'].groupby('dismissal_kind')['dismissal_kind'].count().plot(kind='bar')
df_runs[df_runs.player_dismissed == 'CH Gayle'].groupby('bowler')['bowler'].count().plot(figsize=(20,10),kind='bar')
fig, ax = plt.subplots(figsize=(25,10))
dfgrp = df_runs[df_runs.batsman.isin(['MS Dhoni','CH Gayle','Yuvraj Singh','G Gambhir'])].groupby(['batsman','batsman_runs']).agg({'batsman_runs': 'count'})
dfgrp.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack().plot(ax=ax,kind='bar')
df1 = pd.DataFrame({'wickets':df_runs[df_runs.player_dismissed.notna()].groupby('bowler')['bowler'].count()}).reset_index()
df1= df1.sort_values('wickets',ascending=False)
print(df1[0:10])
df1[0:10].groupby('bowler')['wickets'].sum().plot(kind='barh',figsize=(20,10))
df_runs[df_runs.bowler == 'SL Malinga'].groupby('dismissal_kind')['dismissal_kind'].count().plot(kind='bar')
df_all = df_runs.join(df,on='match_id')
df_all.head()
df_all.shape
df_all.venue.unique()
fig, ax = plt.subplots(figsize=(20,10))
# use unstack()
df_all[df_all.bowler.isin( ['SL Malinga','Harbhajan Singh', 'Z Khan','R Ashwin'])].groupby(['bowler','dismissal_kind']).count()['match_id'].unstack().plot(ax=ax,kind='bar')
fig, ax = plt.subplots(figsize=(20,10))
# use unstack()
df_all[df_all.venue.isin( ['M Chinnaswamy Stadium','Rajiv Gandhi International Stadium, Uppal', 'Wankhede Stadium','Eden Gardens'])].groupby(['venue','dismissal_kind']).count()['match_id'].unstack().plot(ax=ax,kind='bar')
fig, ax = plt.subplots(figsize=(25,10))
# use unstack()
dfgrp = df_all[df_all.venue.isin( ['OUTsurance Oval','Rajiv Gandhi International Stadium, Uppal', 'Barabati Stadium','Eden Gardens'])].groupby(['venue','batsman_runs']).agg({'batsman_runs': 'count'})
dfgrp.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack().plot(ax=ax,kind='bar')
fig, ax = plt.subplots(figsize=(25,10))
# use unstack()
dfgrp = df_all[df_all.venue.isin( ['OUTsurance Oval','Rajiv Gandhi International Stadium, Uppal', 'Barabati Stadium','Eden Gardens'])].groupby(['venue','dismissal_kind']).agg({'dismissal_kind': 'count'})
dfgrp.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack().plot(ax=ax,kind='bar')
