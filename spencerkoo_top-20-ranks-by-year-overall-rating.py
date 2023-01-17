import sqlite3

import pandas as pd

import numpy as np

import datetime



# Reading in the data from the SQLite database file

db = '../input/database.sqlite'

connect = sqlite3.connect(db)

query = "SELECT name FROM sqlite_master WHERE type = 'table';"

pd.read_sql(query, connect)
# Put data into dataframes

query = "SELECT * FROM player;"

players_df = pd.read_sql(query, connect)



query = "SELECT * FROM player_attributes"

player_stats_df = pd.read_sql(query, connect)



# Set options so I can see more of the dataframe

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 200)
# Merge the player and player_attribute data

df = players_df.merge(player_stats_df, how='inner', on='player_api_id')



# Calculate the age of the players

df['birthday'] = pd.to_datetime(df['birthday'])

df['date'] = pd.to_datetime(df['date'])

df['age'] = df['date'] - df['birthday']



# Drop unnecessary columns

df.drop(['id_x', 'id_y', 'player_fifa_api_id_x', 'player_fifa_api_id_y'], 1, inplace=True)

ratings_df = df[['player_api_id', 'player_name', 'age', 'height', 'weight', 'date',

                 'overall_rating', 'potential']]



# Drop players without any rating

ratings_df = ratings_df.drop(ratings_df[ratings_df['overall_rating'].isnull()].index)



# Sorting by rating rather than age

# I will later group by the year and want to have the highest rating per player when I

# remove the duplicate players within each year

ratings_df.sort_values(['player_name', 'player_api_id', 'overall_rating',

                        'potential'], ascending=[True, True, False, False], inplace=True)



# Change the date to just the year

ratings_df['date'] = ratings_df['date'].apply(lambda x: x.year)
# Grouping the players by the year

group = ratings_df.groupby('date')



# Dropping the duplicate player entries per year

ratings_df_unique = group.apply(lambda x: x.drop_duplicates(subset = 'player_api_id', keep = 'first'))



# Grouping the df again into another df group object

group = ratings_df_unique.groupby('date')
# Adding a rating rank var

# Logic here is that if overall rating ties are not taken into account, then the data would

# be biased in an arbitrary manner for regression/prediction later on

# For example, if you have the data ranked by rating and then alphabetically, then those players

# at the end of the alphabet will be left out of the top 20, even if they have the same

# overall rating as those players with names lower in the alphabet

ratings_df_unique['rating_rank'] = group['overall_rating'].rank(ascending=False)
# Return the top 20 overall rating ranks by year

group.apply(lambda x: x.drop(x[x['rating_rank'] > 20].index).sort_values(

    by=['rating_rank', 'potential'], ascending=[True, False]).reset_index(drop=True))