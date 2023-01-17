# import packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 as lite

import datetime as dt

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

database = '../input/database.sqlite'

conn = lite.connect(database)



# get best players from database

query = 'SELECT player_api_id, overall_rating, potential, acceleration FROM Player_Attributes'

df_pot = pd.read_sql(query, conn)



# determine name based on id

query = 'SELECT player_api_id, player_name, birthday FROM Player'

df_name = pd.read_sql(query, conn)



# convert the birthday -> age

y = dt.datetime.now()

age = y-pd.DatetimeIndex(df_name['birthday'])

df_name['age'] = age.total_seconds()/(365*86400)

df_name = df_name.drop('birthday', 1)
# merge dataframes

df_combo = pd.merge(df_pot, df_name, on='player_api_id')



# remove missing data

df_combo = df_combo.dropna(how='any')



# check the top players

df_combo = df_combo.sort_values(by='overall_rating', ascending=False)

df_combo = df_combo.drop_duplicates('player_name')

df_combo['growth'] = df_combo['potential'] - df_combo['overall_rating']

df_combo.head()
# dirty plot to check relation

df_combo.plot(kind='hexbin', x='age', y='growth',C='overall_rating',gridsize=30,cmap=plt.cm.RdYlGn)

plt.title('Investigating Growth with Age')

plt.xlim([15, 50])

plt.ylim([-20, 20])
# dirty plot to check relation

df_combo.plot(kind='hexbin', x='age', y='growth',C='acceleration',gridsize=30,cmap=plt.cm.RdYlGn)

plt.title('Investigating Growth with Age')

plt.xlim([15, 50])

plt.ylim([-20, 20])