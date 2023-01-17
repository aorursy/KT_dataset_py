# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



df_player_of_the_week = pd.read_csv("../input/NBA_player_of_the_week.csv")
df_seasons_in_league_domination = df_player_of_the_week.groupby(['Seasons in league'])['Real_value'].sum().reset_index()

plt.figure(figsize=(15,5))

plt.bar(pd.to_numeric(df_seasons_in_league_domination['Seasons in league']),df_seasons_in_league_domination['Real_value'])

plt.xticks(pd.to_numeric(df_seasons_in_league_domination['Seasons in league']))

plt.xlabel('Seasons in league')

plt.ylabel('Real value')

plt.title('Domination by seasons in league')

plt.show()
df_seasons_in_league_domination_over_seasons = df_player_of_the_week.groupby(['Season short','Seasons in league'])['Real_value'].sum().reset_index()



plt.figure(figsize=(15,5))

plt.scatter(df_seasons_in_league_domination_over_seasons['Season short'], 

            df_seasons_in_league_domination_over_seasons['Seasons in league'],

            s=df_seasons_in_league_domination_over_seasons['Real_value']**2)

plt.xlabel('Season')

plt.ylabel('Seasons in league')

plt.title('Regular Season domination by seasons in league')

plt.show()
# get the total points by player

df_players_domination = df_player_of_the_week.groupby(['Player'])[

    'Real_value'].sum().reset_index().rename(columns={'Real_value': 'Total_real_value'})

# get top 10 players

df_players_domination = df_players_domination.nlargest(10,'Total_real_value')

df_player_of_the_week_for_top = df_player_of_the_week.groupby(['Player','Season short','Season','Seasons in league'])[

      'Real_value'].sum().reset_index()

df_players_domination_by_season_in_league = pd.merge(df_players_domination, df_player_of_the_week_for_top, how='left', 

                                                     on=['Player'], copy=True)    

# lets see who are the top 10 players

df_players_domination
df_player_of_the_week_for_top_bar = df_players_domination_by_season_in_league.groupby(['Seasons in league'])['Real_value'].sum().reset_index()

plt.figure(figsize=(15,5))

plt.bar(pd.to_numeric(df_player_of_the_week_for_top_bar['Seasons in league']),

        df_player_of_the_week_for_top_bar['Real_value'])

plt.xticks(pd.to_numeric(df_player_of_the_week_for_top_bar['Seasons in league']))

plt.xlabel('Seasons in league')

plt.ylabel('Real value')

plt.title('Top 10 players domination by seasons in league')

plt.show()
plt.figure(figsize=(15,5))

plt.scatter(df_players_domination_by_season_in_league['Season short'], 

            df_players_domination_by_season_in_league['Seasons in league'],

            s=df_players_domination_by_season_in_league['Real_value']**3)

plt.xlabel('Season')

plt.ylabel('Seasons in league')

plt.title('Top 10 players domination by seasons in league')

plt.show()
df_players_domination_by_season_in_league[df_players_domination_by_season_in_league['Seasons in league'] == 0].groupby(

    ['Player','Season'])['Real_value'].sum().reset_index().sort_values('Real_value',ascending= False)
df_players_domination_by_season_in_league[df_players_domination_by_season_in_league['Seasons in league'] == 12].groupby(

    ['Player','Season'])['Real_value'].sum().reset_index().sort_values('Real_value',ascending= False)