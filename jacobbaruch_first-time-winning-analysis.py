# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

df_player_of_the_week = pd.read_csv("../input/NBA_player_of_the_week.csv")



# Any results you write to the current directory are saved as output.
# get first player of the week appearance of each player

df_player_of_the_week['Date'] = pd.to_datetime(df_player_of_the_week.Date)

first_win = df_player_of_the_week.groupby('Player')['Date'].agg(['first'])

first_win.reset_index(level=0,inplace=True)

# show dataframe content and num of rows

first_win.info()
# Attach details for each player

first_win_full_df = pd.merge(first_win, df_player_of_the_week, how='inner', left_on=['Player','first'], right_on=['Player','Date'])



first_win_full_df.info()
df_first_time_by_seasons_in_leg = first_win_full_df.groupby(['Seasons in league'])['Real_value'].sum().reset_index()

plt.figure(figsize=(8,3.7))

plt.bar(pd.to_numeric(df_first_time_by_seasons_in_leg['Seasons in league']),df_first_time_by_seasons_in_leg['Real_value'])

plt.xticks(pd.to_numeric(df_first_time_by_seasons_in_leg['Seasons in league']))

plt.xlabel('Seasons in league')

plt.ylabel('Real value')

plt.title('First award winning - by season in league')
# checking out who won after that period of the time for the first time.

first_win_full_df[['Player','Seasons in league','Season short']].sort_values(by=['Seasons in league'],ascending=False).head(4)