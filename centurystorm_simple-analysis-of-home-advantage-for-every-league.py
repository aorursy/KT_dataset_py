#######################

# Simple analysis of home advantage for every league

# Lars Tijssen | lars.tijssen@essl.org

# database: European Soccer League

# 2017-05-31

#######################



%matplotlib inline



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3
# load data

path = '../input/database.sqlite'

with sqlite3.connect(path) as con:

    countries = pd.read_sql_query("SELECT * from Country", con)

    matches = pd.read_sql_query("SELECT * from Match", con)

    leagues = pd.read_sql_query("SELECT * from League", con)

    teams = pd.read_sql_query("SELECT * from Team", con)
# Sanitize the data



## select the id-column as the index of the dataframe

matches = matches.set_index('id')



## map the actual name to the id 

# league

league_id_map = leagues.set_index('id').to_dict()['name']

matches['league_id'] = matches['league_id'].map(league_id_map)



# country

country_id_map = countries.set_index('id').to_dict()['name']

matches['country_id'] = matches['country_id'].map(country_id_map)



# teams

team_id_map = teams.set_index('team_api_id').to_dict()['team_long_name']

matches['home_team_api_id'] = matches['home_team_api_id'].map(team_id_map)

matches['away_team_api_id'] = matches['away_team_api_id'].map(team_id_map)



## convert the date column to python datetime

matches.date = pd.to_datetime(matches.date)
# calculate the mean goals per league

mean_goals = matches[['league_id','home_team_goal','away_team_goal']].groupby('league_id').mean()
# calculate the mean goals per league as percentage from the total

total_goals = mean_goals['home_team_goal'] + mean_goals['away_team_goal']

mean_goals['home_team_goal_percent'] = 1e2 * mean_goals['home_team_goal'] / total_goals

mean_goals['away_team_goal_percent'] = 1e2 * mean_goals['away_team_goal'] / total_goals



# make a subset of 

mean_goals_percent = mean_goals[['home_team_goal_percent','away_team_goal_percent']]
mean_goals_percent
#### plot the mean goals per league as a horizontal bar chart using pandas plotting

mgp_plot = mean_goals_percent.plot(kind='barh', stacked=True, legend = False,

                                   fontsize = 12, width = 0.8, xticks = np.arange(0,100.1,10))



mgp_plot.set_xlabel('Percent', fontsize = 14)

mgp_plot.set_ylabel('League', fontsize = 14)

mgp_plot.set_title('Ratio between home goals and away goals for each League in percent', fontsize = 16)



mgp_plot.vlines(50, ymin = -0.5, ymax = 10.5, linewidth = 0.5, linestyle = '--')



# plot numbers in plot

k = 0

for index, row in mean_goals_percent.iterrows():

    

    k1 = row['home_team_goal_percent'].round(2)

    k2 = row['away_team_goal_percent'].round(2)

    

    mgp_plot.text(x = 25, y = k, s = k1, color = 'white', fontsize = 14, ha = 'center', va = 'center')

    mgp_plot.text(x = 80, y = k, s = k2, color = 'white', fontsize = 14, ha = 'center', va = 'center')

    k += 1

    

plt.legend(bbox_to_anchor=(1.45,1), fontsize = 12)