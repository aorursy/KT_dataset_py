import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dfm = pd.read_csv("../input/ipldata/matches.csv")
# First five rows

dfm.head()
# Shape of the dataset

print(dfm.shape)
# list of all columns

print(dfm.columns)
# Total no of seasons

dfm['season'].unique()
# Total number of wiining by each team across all the seasons

dfm.groupby(by=['winner'])['id'].count()
# Total number of wiining by each team across all the seasons without groupby

dfm['winner'].value_counts
a =dfm['winner'].value_counts()

#dfm['winner'].value_counts().plot(kind='bar',color='r')   This is using pandas

#plt.show()



sns.barplot(x=a.index,y=a.values,data=dfm)   # This is using seaborn

plt.xticks(rotation=90,fontsize=10)
#Total Number of matches played by each team

b = dfm['team1'].value_counts() + dfm['team2'].value_counts()

b.sort_values(ascending = False)
#Total Number of matches played by each team in bar graph

sns.barplot(x=b.index,y=b.values)

plt.xticks(rotation=90,fontsize=10)
# Percentage winning of each team

p_winning  = a/b *100

p_winning.sort_values(ascending = False)
# Percentage winning of each team with graph

p_winning.sort_values(ascending = False).plot(kind='bar')

plt.show()
# Top 10 man of the match

dfm['player_of_match'].value_counts().head(20)
dfm['player_of_match'].value_counts().head(20).plot(kind='bar')

plt.xticks(rotation=90,fontsize=10)

plt.show()
# Maximum win by run

dfm[dfm['win_by_runs']==dfm['win_by_runs'].max()]
# Maximum win by wickets

dfm[dfm['win_by_wickets']==dfm['win_by_wickets'].max()]
# No of matches plaed in cities

dfm['city'].value_counts()
dfm['city'].value_counts().plot(kind='bar')

plt.show()
# How many matches were played in very season

dfm.groupby(by='season')['id'].count()
dfm.groupby(by='season')['id'].count().plot(kind='bar')
dfm.groupby(by='winner')['id'].count()
dfm[(dfm['winner']=='Chennai Super Kings') & (dfm['toss_winner']=='Chennai Super Kings')].count()['id']/dfm[dfm['winner']=='Chennai Super Kings']['id'].count()*100
dfm[(dfm['winner']=='Chennai Super Kings') & (dfm['toss_winner']!='Chennai Super Kings')].count()['id']/dfm[dfm['winner']=='Chennai Super Kings']['id'].count()*100
# winning percentage of each team when they win the toss



for i in dfm['winner'].unique():

    print(i,':',dfm[(dfm['winner']==i) & (dfm['toss_winner']==i)].count()['id']/dfm[dfm['winner']==i]['id'].count()*100)
# winning percentage of each team when they lose the toss



for i in dfm['winner'].unique():

    print(i,':',dfm[(dfm['winner']==i) & (dfm['toss_winner']!=i)].count()['id']/dfm[dfm['winner']==i]['id'].count()*100)
# Matches played in stadiums

dfm['venue'].value_counts().plot(kind='bar')
mvc =dfm[((dfm['team1']=='Chennai Super Kings') & (dfm['team2']=='Mumbai Indians')) | ((dfm['team1']=='Mumbai Indians') & (dfm['team2']=='Chennai Super Kings'))]
mvc
mvc['winner'].value_counts()
sns.countplot(mvc['winner'])
sns.countplot(mvc['winner'],hue=mvc['toss_decision'])
sns.countplot(mvc['winner'],hue=mvc['toss_winner'])
mvc.groupby(by=['venue','winner'])['id'].count()
# total wins per session per teams

for i in dfm['season'].unique():

    print(i,"\n")

    print(dfm[dfm['season']==i]['winner'].value_counts())
dfm.head()
# Most no of umpiring

dfm['umpire1'].value_counts().head(10)
o = dfm[(dfm['venue']=='Wankhede Stadium') & (dfm['toss_decision']=='field')]
o
ans = o[o['win_by_wickets']>0]['id'].count()/o['id'].count()*100

print("What is the win percentage of a team batting second at Wankhede Stadium during 2008 to 2019",ans)
dfm['dl_applied'].value_counts()
dfm[dfm['dl_applied']==1]['winner'].value_counts()
dfm[dfm['dl_applied']==1]['winner'].value_counts().plot(kind='bar')