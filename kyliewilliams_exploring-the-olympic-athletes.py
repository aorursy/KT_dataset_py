# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



results = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

regions = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')



data = pd.merge(results, regions, on='NOC', how='left')

data.info()

smushedtosinglesport = (data[['ID','Name','Sport', 'Season']])

smushedtosinglesport = (smushedtosinglesport.groupby(['ID'])).filter(lambda x: (x.Season.nunique() > 1))

smushedtosinglesport.tail()
sns.violinplot(y='Sport', data=smushedtosinglesport.tail(50))
athletes = groupedByID.size()

most_represented_athletes = pd.DataFrame({'ID':athletes.index, 'Count':athletes.values})

#Ranking the count of medalled sport branches in descending order.

most_represented_athletes.sort_values(['Count', 'ID'], ascending=[False, True], inplace=True)



most_represented_athletes.head()
super_athlete = data.loc[(data.ID == 77710)]

super_athlete.head()
sns.countplot(x='Year',data=super_athlete,palette=sns.cubehelix_palette(8, start=.5, rot=-.75))

plt.title('Events entered by Robert Tait McKenzie')

plt.ylabel('Number of Events')
robert_medals = super_athlete.loc[pd.notnull(super_athlete.Medal)]

robert_medals
most_games = groupedByID.Games.nunique()

most_games_athletes = pd.DataFrame({'ID':most_games.index, 'Count':most_games.values})

#Ranking the count of medalled sport branches in descending order.

most_games_athletes.sort_values(['Count', 'ID'], ascending=[False, True], inplace=True)

greatereq_than_eight = most_games_athletes[most_games_athletes.Count >= 8]



athlete_at_least_eight_games = data.loc[data.ID.isin(greatereq_than_eight.ID)]
sns.countplot(x='Sport',data=athlete_at_least_eight_games)