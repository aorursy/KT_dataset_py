#cd /Users/akash-5162/Desktop/DS/Git/Projects/Kaggle Datasets
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
football_result=pd.read_csv('../input/womens-international-football-results/results.csv',header=0,parse_dates=[0])

football_result.head()
football_result.info()
football_result.describe()
football_result[['home_score','away_score']].plot(kind='box',figsize=(10,10),title='distribution of Away and Home goals')
football_result['tournament'].value_counts()
football_result.groupby('tournament')['date'].count().sort_values(ascending=False)[0:10].plot(kind='bar',title='Top 10 contributing leagues',figsize=(15,10))
football_result.groupby(football_result['date'].dt.year)['tournament'].count().plot(kind='bar',figsize=(15,10))
football_result['home_team_win']=football_result['home_score']>football_result['away_score']
football_result.groupby('home_team')['home_team_win'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',figsize=(10,10),title='top ten teams and their home match wins')
football_result.groupby('home_team')['home_score'].sum().sort_values(ascending=False)[0:20].plot(kind='bar',figsize=(15,10),title='Top 20 teams based on their total home game goals')
football_result['away_team_win']=football_result['home_score']<football_result['away_score']
football_result.head()
football_result.groupby('away_team')['away_team_win'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',figsize=(10,10),title='Top 10 countries to have crushed the home team')
football_result.groupby('away_team')['away_score'].sum().sort_values(ascending=False)[0:20].plot(kind='bar',title='Top 20 teams based on their goals in away matches',figsize=(15,10))
football_result['month']=football_result.date.dt.month
x=[0,1,2,3,4,5,6,7,8,9,10,11]

label=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']



football_result.groupby('month')['home_team'].count().plot(kind='bar',figsize=(10,10),title='Month specific games count')

plt.xticks(x,label,rotation='horizontal')
neutral_games=football_result[football_result['neutral']==True]

neutral_games.head()
neutral_games['city'].count()



print('There are about 1839 games played in neutral venues')
neutral_games['home_team_win']=neutral_games['home_score']>neutral_games['away_score']
neutral_games.tail()
neutral_games.groupby('home_team')['home_team_win'].sum().sort_values(ascending=False)[0:20].plot(kind='bar',figsize=(15,10),title='Top 20 teams and their wins in neutral venues')
plt.hist(neutral_games['home_score']-neutral_games['away_score'],bins=10)
neutral_games[['home_score','away_score']].plot(kind='hist',bins=10,subplots=True,figsize=(10,10))
neutral_games.groupby(neutral_games['date'].dt.year)['away_team'].count().plot(kind='bar',figsize=(15,10),title='Trend of neutral games counts every year')
neutral_games.groupby('country')['city'].count().sort_values(ascending=False)[0:10].plot(kind='bar',figsize=(10,10),title='Top 10 countries to host neutral games')
neutral_games[neutral_games['country']=='Portugal']['city'].agg({'city':pd.Series.nunique})



#neutral_games[neutral_games['country']=='Portugal']['city'].agg({'city':pd.Series.nunique})
neutral_games[neutral_games['country']=='Portugal']['city'].value_counts().plot(kind='bar',figsize=(10,10))
neutral_games[np.logical_and(neutral_games['country']=='Portugal', neutral_games['home_team_win']==True)]['home_team'].value_counts().sort_values(ascending=False)[0:10].plot(kind='bar')
l=np.logical_and(neutral_games['country']=='Portugal', neutral_games['home_team']=='Norway')

m=np.logical_and(l,neutral_games['home_team_win']==True)

neutral_games[m]['city'].value_counts()
neutral_games.groupby('tournament')['city'].count().sort_values(ascending=False).plot(kind='bar',figsize=(10,10),title='Tournaments and their respective number of games in neutral venues')