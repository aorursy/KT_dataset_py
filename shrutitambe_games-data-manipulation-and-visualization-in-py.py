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
games[['playingtime','total_comments']].corr() # No correlation.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)
games.dropna(inplace=True)
sns.distplot(games["average_rating"])
sns.jointplot(games["minage"], games["average_rating"])
sns.pairplot(games[["playingtime", "minage", "average_rating"]])

sns.stripplot( games["type"], games["playingtime"], jitter= True)
sns.lmplot(x= "playingtime", y= "average_rating", data = games[games["playingtime"]<500])