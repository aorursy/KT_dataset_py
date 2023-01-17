import numpy as np 
import pandas as pd 
import os
import datetime
print(os.listdir("../input"))
import plotly
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
import itertools
games = pd.read_csv('../input/games.csv')
meta_keys = [list(meta_keys) for meta_keys in games['meta'].fillna('[]').apply(literal_eval).apply(lambda x: x.keys() if isinstance(x, dict) else [])]
addtional_columns = set(itertools.chain.from_iterable(meta_keys))
for column in addtional_columns:
    games[column.replace('boardgame', '')] = games['meta'].fillna('[]').apply(literal_eval).apply(lambda x: x[column].split('|') if column in x.keys() and isinstance(x, dict) else [])
games = games.drop(['meta'], axis=1)
games.head(10)
# Duplicate rows for mechanics
mechanics = games.apply(lambda x: pd.Series(x['mechanic']),axis=1).stack().reset_index(level=1, drop=True)
mechanics.name = 'mechanic'
mechanics_df = games.drop('mechanic', axis=1).join(mechanics)
mechanics_df['mechanic'] = mechanics_df['mechanic'].astype("category")
high_rated_games = mechanics_df[(mechanics_df['users_rated']>200) & (mechanics_df['average_rating']>float(7.5)) & (mechanics_df['type'] == 'boardgame')]
len(high_rated_games)
# For each year I want to see what percentage of highly rated games used each mechanic
mechanics_by_year = high_rated_games[['yearpublished','mechanic']]
# mechanics_by_year = mechanics_by_year.groupby(['yearpublished', 'mechanics']).count()
# mechanics_by_year = mechanics_by_year.reset_index()
# mechanics_by_year.rename(columns={'id': 'count'}, inplace=True)
mechanics_by_year['yearpublished'] = mechanics_by_year['yearpublished'].astype(int)
mechanics_by_year = mechanics_by_year[mechanics_by_year['yearpublished'] > 0]

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

g = sns.catplot(x="mechanic", y="yearpublished", data=mechanics_by_year[mechanics_by_year["yearpublished"] >=1968], height=30, kind="box")
g.set_xticklabels(rotation=90)
most_popular_designers = games[(games['users_rated']>200) & (games['type'] == 'boardgame')].sort_values(by=['average_rating'], ascending=False).head(500)
desginers = most_popular_designers.apply(lambda x: pd.Series(x['designer']),axis=1).stack().reset_index(level=1, drop=True)
desginers.name = 'designer'
most_popular_designers = most_popular_designers.drop('designer', axis=1).join(desginers)
most_popular_designers['designer'] = most_popular_designers['designer'].astype("category")
most_popular_designers = most_popular_designers[['id', 'designer']]
most_popular_designers.groupby('designer').count().sort_values(by='id', ascending=False)