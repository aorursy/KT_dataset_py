# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
fifa = pd.read_csv('../input/data.csv')

fifa.shape
fifa.columns
fifa.head()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10,6))

players_nation = fifa['Nationality'].value_counts()[:10]

sns.barplot(players_nation.index,players_nation.values)

plt.xticks(rotation=40)
#Players with highest and lowest overall rating

top_5_players = fifa[['Name','Overall']].sort_values(by='Overall',ascending=False)[:5]

bottom_5_players = fifa[['Name','Overall']].sort_values(by='Overall')[:5]
top_5_players,bottom_5_players
fifa.nunique()
#Combined rating of players in top 10 clubs



top_10_ratedclubs = fifa.groupby(['Club'])['Overall'].sum().sort_values(ascending=False)[:10]

top_10_ratedclubs
#Top 10 clubs with most diversity in players based on nationality

diversity = fifa.groupby(['Club'])['Nationality'].nunique().nlargest(10)

sns.barplot(diversity.index,diversity.values,palette='plasma')

plt.xticks(rotation=90)

plt.title('Diversity based on Nationality')
#Relation between Overall rating and wages

#fifa['Wage']= fifa['Wage'].str.replace('[^0-9]','').astype(int)*1000

rating_wages = pd.DataFrame(fifa[['Overall','Wage','Position']])

plt.figure(figsize=(10,8))

sns.scatterplot(x='Overall',y='Wage',hue='Preferred Foot',data=fifa)

#sns.scatterplot(fifa['Overall'],fifa['Wage'].str.replace('[^0-9]','').astype(int)*1000,data=fifa)
'Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes'
attack = pd.DataFrame(fifa[['Position','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle']])

plt.figure(figsize=(12,10))

sns.heatmap(attack.corr())
#Average characteristics of attacking players in each position

attacking_players = fifa[fifa.Position.isin(['LW','RW','ST','CAM','LAM','RAM'])].groupby(['Position'])['Dribbling','BallControl','Acceleration','SprintSpeed','Balance','ShotPower'].mean()

attacking_players.reset_index(inplace=True)

attacking_players
#Average characteristics of defending players in each position

defending_players = fifa[fifa.Position.isin(['CDM','CM','LM','RM','LB','RB','CB'])].groupby(['Position'])['Strength','Aggression','Interceptions','Composure','Marking','StandingTackle','SlidingTackle'].mean()

defending_players.reset_index(inplace=True)

defending_players