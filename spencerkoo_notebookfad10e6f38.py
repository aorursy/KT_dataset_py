import sqlite3

import pandas as pd

import datetime
db = '../input/database.sqlite'

connect = sqlite3.connect(db)

query = "SELECT name FROM sqlite_master WHERE type = 'table';"

pd.read_sql(query, connect)
query = "SELECT * FROM player;"

players_df = pd.read_sql(query, connect)

players_df.info()
query = "SELECT * FROM player_attributes;"

player_stats_df = pd.read_sql(query, connect)
df = players_df.merge(player_stats_df, how='inner', on='player_api_id')
df.drop(['id_x', 'id_y', 'player_fifa_api_id_x', 'player_fifa_api_id_y'], 1, inplace=True)
df['birthday'] = pd.to_datetime(df['birthday'])

df['date'] = pd.to_datetime(df['date'])

df['age'] = df['date'] - df['birthday']
ratings_df = df

ratings_df = ratings_df.drop(ratings_df[ratings_df['overall_rating'].isnull()].index)
ratings_df.sort_values(['player_name', 'player_api_id', 'overall_rating'], ascending=[True, True, False], inplace=True)
ratings_df['date'] = ratings_df['date'].apply(lambda x: x.year)
group = ratings_df.groupby('date')
ratings_df = ratings_df.drop(ratings_df[ratings_df['gk_reflexes'] > 50].index)
ratings_df['proxy'] = ratings_df[9:-5].mean(0)
ratings_df