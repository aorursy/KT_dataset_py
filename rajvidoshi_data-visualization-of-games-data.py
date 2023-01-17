import pandas as pd
games = pd.read_csv('../input/games-data/games.csv')
games.head()
games.info()
games['playingtime'].mean()
games['total_comments'].max()
games[games['id']==1500]['name']
games[games['id']==1500]['yearpublished']
games[games['total_comments']== games['total_comments'].max()]
games[games['total_comments']== games['total_comments'].min()]
games.groupby('type').mean()['minage']
games['id'].nunique()
games['type'].value_counts()
games[['playingtime','total_comments']].corr()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

%matplotlib inline
games.dropna(inplace=True)

games.info()
sns.distplot(games['average_rating']);
games1 = games[games['average_rating']!=0]

games1.info()
sns.jointplot(games1['minage'], games1['average_rating']);
sns.pairplot(games1[['playingtime', 'minage', 'average_rating']]);
sns.stripplot(games1['type'], games1['playingtime'], jitter=True);
sns.regplot(x="playingtime", y="average_rating", data=games1[games1['playingtime'] < 500]);