 %matplotlib inline



import sqlite3

import pandas as pd

import seaborn as sns

import datetime

import numpy as np

import matplotlib.pyplot as plt



database = '../input/database.sqlite'

conn = sqlite3.connect(database)
query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)
players_df = pd.read_sql_query("SELECT * from Player", conn)

player_stats_df = pd.read_sql_query("SELECT * from Player_Attributes", conn)
players_df.head()
players_df.info()
player_stats_df.head()
player_stats_df.info()
df = players_df.merge(player_stats_df, how='inner', on='player_api_id')
df['birthday'] = pd.to_datetime(df['birthday'])

df['anonac'] = df['birthday'].dt.year

df['date'] = pd.to_datetime(df['date'])

df['anocor'] = df['date'].dt.year

df['anos'] = df['anocor'] - df['anonac']
df.drop(['id_x', 'id_y', 'player_fifa_api_id_x', 'player_fifa_api_id_y'], 1, inplace=True)

ratings_df = df[['player_api_id', 'player_name', 'anos', 'date',

                 'overall_rating', 'potential']]
ratings_df = ratings_df.drop(ratings_df[ratings_df['overall_rating'].isnull()].index)



ratings_df.sort_values(['player_name', 'player_api_id', 'overall_rating',

                        'potential'], ascending=[True, True, False, False], inplace=True)



ratings_df['date'] = ratings_df['date'].apply(lambda x: x.year)
group = ratings_df.groupby('anos')

ratings_df_unique = group.apply(lambda x: x.drop_duplicates(subset = 'player_api_id', keep = 'first'))

group = ratings_df_unique.groupby('anos')

ratings_df_unique['rating_rank'] = group['overall_rating'].rank(ascending=False)
losmejoresporedad = group.apply(lambda x: x.drop(x[x['rating_rank'] > 1].index).sort_values(

    by=['rating_rank', 'potential'], ascending=[True, False]).reset_index(drop=True))

losmejoresporedad