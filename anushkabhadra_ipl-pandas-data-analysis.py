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
#1.How to import csv files

match=pd.read_csv('/kaggle/input/ipl/matches.csv')
match
#Two important data structures of pandas are

#-->DataFrames

#-->Series
type(match)
match.head()
match.head(1)
match.head(3)
match.tail()
match.tail(1)
match.tail(3)
match.head()
match['city']
match.head()

match['city']
match['city']
mask=match['city']=='Mumbai'

match[mask]['player_of_match'].value_counts().head(1).index[0]

mask=match['season']>=2015

match[mask]['winner'].value_counts().head(1)
mask1=match['city']=='Hyderabad'

mask2=match['season']>=2015
match[mask1 & mask2]
match[mask1 & mask2]['player_of_match'].value_counts().head(1).index[0]
# Plot graphs

# Sort_values

# drop_duplicates



# Problems
import matplotlib.pyplot as plt
match['winner'].value_counts().head().plot(kind='bar')
match['winner'].value_counts().head().plot(kind='barh')
match.head()
match['toss_decision'].value_counts().plot(kind='pie')
match.head()
match.drop_duplicates(subset=['season'])
match.drop_duplicates(subset=['season']).shape
match['toss_decision'].value_counts().head(1).index[0]
match['toss_decision'].value_counts().sum()
(match['toss_decision'].value_counts()/match['toss_decision'].value_counts().sum()*100).head(1)
mask=match['city']=='Mumbai'

match[mask]['city'].value_counts()
match[match['city']=='Kolkata']['player_of_match'].value_counts().head(1)

mask1=match['city']=='Kolkata'

mask2=match['season']==2015

match[mask1&mask2]['player_of_match'].value_counts().head(1)
mask1= (match['team1']=='Royal Challengers Bangalore') & (match['team2']=='Mumbai Indians')

mask2= (match['team1']=='Mumbai Indians') & (match['team2']=='Royal Challengers Bangalore')

new= match[mask1 | mask2]

new.shape[0]

new[new['winner']=='Royal Challengers Bangalore'].shape[0]
new[new['winner']=='Mumbai Indians'].shape[0]
def teamVteam(team1,team2):

    mask1= (match['team1']==team1) & (match['team2']==team2)

    mask2= (match['team1']==team2) & (match['team2']==team1)

    new= match[mask1 | mask2]

    total_matches= new.shape[0]

    won_team1 = new[new['winner']==team1].shape[0]

    won_team2 = new[new['winner']==team2].shape[0]

    

    print("Number of matches",total_matches)

    print("Matches won by {}".format(team1),won_team1)

    print("Matches won by {}".format(team2),won_team2)

    print("Matches drawn",total_matches-(won_team1+won_team2))
teamVteam('Kolkata Knight Riders','Royal Challengers Bangalore')
delivery = pd.read_csv('/kaggle/input/ipl/deliveries.csv')
delivery.shape
delivery.head()
# GroupBy
mask=delivery['batsman']=='RG Sharma'

delivery[mask]['batsman_runs'].sum()
# Top 5 batsman(most no.of runs scored)

batsman= delivery['batsman'].unique()

batsman
delivery.groupby('batsman').sum()['batsman_runs'].sort_values(ascending=False).head()
six=delivery[delivery['batsman_runs']==6]

six.groupby['batsman'].count()['batsman_runs'].sort_values(ascending=False).head()
six.groupby['batsman'].counts()['batsman_runs'].sort_values(ascending=False).head()
delivery[delivery['batsman']=='V Kohli'].groupby('bowling_team').sum()['batsman_runs'].sort_values(ascending=False).head(1).index[0]
def batsmanVbowler(batsman):

    return delivery[delivery['batsman']==batsman].groupby('bowler').sum()['batsman_runs'].sort_values(ascending=False).head(1).index[0]
batsmanVbowler('MS Dhoni')
six=delivery[delivery['batsman_runs']==6]
pt = six.pivot_table(index='over',columns='batting_team',values='batsman_runs',aggfunc='count')

pt
import seaborn as sns
sns.heatmap(pt, cmap='spring')
# how to merge two or more dataframes
match=pd.read_csv('/kaggle/input/ipl/matches.csv')
match
merged=delivery.merge(match,left_on='match_id', right_on='id')
merged[merged['batsman']=='MS Dhoni'].groupby('season').sum()['batsman_runs'].plot(kind='line')
def plotCareerCurve(batsman):

    merged[merged['batsman']==batsman].groupby('season').sum()['batsman_runs'].plot(kind='line')
plotCareerCurve('DA Warner')