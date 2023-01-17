# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
matches = pd.read_csv("../input/FMEL_Dataset.csv")
print(matches.head())
matches.isnull().sum()
matches.info()
division1 = matches[matches['division'] == 1]
division1.drop('division', axis = 1, inplace = True)
division1.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
local_visitor_goals = division1.groupby('season')[['localGoals', 'visitorGoals']].sum()
local_visitor_goals.reset_index(inplace = True)
local_visitor_goals.head()
plt.figure(figsize=(20, 12))
plt.xticks(rotation='vertical')
sns.set_color_codes("pastel")
g1 = sns.barplot(x = 'season', y = 'localGoals', data = local_visitor_goals, label = "Local", color = 'b')
sns.set_color_codes("muted")
g2 = sns.barplot(x = 'season', y = 'visitorGoals', data = local_visitor_goals, label = "Visitor", color = 'b')

for rect in g1.patches:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

    
plt.ylabel("")
plt.xlabel("Total number of goals")
plt.legend()
local_team = division1.groupby('localTeam')['localGoals'].sum()
local_team = pd.DataFrame(local_team)
local_team.head()
plt.figure(figsize=(12, 12))
plt.xticks(rotation='vertical')
g = sns.barplot(x = 'localTeam', y = 'localGoals', data = local_team.sort_values('localGoals').reset_index())
plt.title("Total Local Goals")
for rect in g.patches:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height + 20, '%d' % int(height), ha='center', va='bottom', rotation = 'vertical')
visitor_team = division1.groupby('visitorTeam')['visitorGoals'].sum()
visitor_team = pd.DataFrame(visitor_team)
visitor_team.head()
plt.figure(figsize=(12, 12))
plt.xticks(rotation='vertical')
g = sns.barplot(x = 'visitorTeam', y = 'visitorGoals', data = visitor_team.sort_values('visitorGoals').reset_index())
plt.title("Total Visitor Goals")
for rect in g.patches:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height + 20, '%d' % int(height), ha='center', va='bottom', rotation = 'vertical')
seasons = list(division1['season'].unique())
teams = list(division1['localTeam'].unique())
season_local_visitor = division1.groupby(['season', 'localTeam', 'visitorTeam'])[['localGoals', 'visitorGoals']].sum()
season_local_visitor.head()

season_local_visitor['difference'] = season_local_visitor['localGoals'] - season_local_visitor['visitorGoals']
season_local_visitor.head()

# 1 for win, -1 for loss, 0 for draw (or tie)
season_local_visitor['localWin'] = season_local_visitor['difference'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
season_local_visitor.reset_index(inplace = True)
season_local_visitor.head()
seasonBefore1995 = seasons[0:list(season_local_visitor['season'].unique()).index('1994-95')+1]
seasonBefore1995
season_local_visitor['localPoints'] = season_local_visitor.apply(lambda x: 2 if (x['localWin'] == 1 and x['season'] in seasonBefore1995) 
                                                                 else (3 if (x['localWin'] == 1 and x['season'] not in seasonBefore1995)
                                                                 else (1 if x['localWin'] == 0 else 0)), axis = 1)
#season_local_visitor
season_local_visitor['visitorPoints'] = season_local_visitor.apply(lambda x: 2 if (x['localPoints'] == 0 and x['season'] in seasonBefore1995)
                                                                   else (3 if (x['localPoints'] == 0 and x['season'] not in seasonBefore1995)
                                                                   else (1 if x['localPoints'] == 1 else 0)), axis = 1)
season_local_visitor.head()
# season_local_visitor.reset_index(inplace = True)
# season_local_visitor.head(15)
total_localPoints = season_local_visitor.groupby(['season', 'localTeam'])[['localPoints']].sum()
total_localPoints.head()
total_visitorPoints = season_local_visitor.groupby(['season', 'visitorTeam'])[['visitorPoints']].sum()
total_visitorPoints.head()
total_points = pd.concat([total_localPoints, total_visitorPoints], axis = 1)
total_points['totalPoints'] = total_points['localPoints'] + total_points['visitorPoints']
total_points.head(21)
max_df = pd.DataFrame(columns = ['Season', 'Max Total Points', 'Max Total Points Team'])
for season in seasons:
    seasonSummary = pd.DataFrame(total_points.loc[season])
    max_points = seasonSummary['totalPoints'].max()
    max_points_team = seasonSummary['totalPoints'].idxmax()
    max_points_ALL_teams = list(seasonSummary.loc[seasonSummary['totalPoints'] == max_points].index)#This returns a list of the teams that have the max points
    if len(max_points_ALL_teams) == 2:#This conditional statement is to break a tie between two teams that have the same max number of points
        #print(season, '\n', max_points_ALL_teams[0: 2])
        first_match = pd.DataFrame(season_local_visitor[(season_local_visitor['season'] == season) & 
                                   (season_local_visitor['localTeam'] == max_points_ALL_teams[0]) & 
                                   (season_local_visitor['visitorTeam'] == max_points_ALL_teams[1])
                                  ])
        second_match = pd.DataFrame(season_local_visitor[(season_local_visitor['season'] == season) & 
                                   (season_local_visitor['localTeam'] == max_points_ALL_teams[1]) & 
                                   (season_local_visitor['visitorTeam'] == max_points_ALL_teams[0])
                                  ])
        
        first_second_matches = pd.concat([first_match, second_match])
        if (first_second_matches[first_second_matches['localTeam'] == max_points_ALL_teams[0]]['difference'].item()>
             first_second_matches[first_second_matches['localTeam'] == max_points_ALL_teams[1]]['difference'].item()):
            max_points_team = max_points_ALL_teams[0]
        else:
            max_points_team = max_points_ALL_teams[1]
        
    max_df = max_df.append({'Season': season,
                            'Max Total Points': max_points,
                            'Max Total Points Team': max_points_team,
                           }, 
                            ignore_index=True)
max_df
plt.figure(figsize = (20, 8))
plt.xticks(rotation = 'vertical')
g = sns.barplot(x = 'Season', y = 'Max Total Points', data = max_df)
for i, rect in enumerate(g.patches):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.text(rect.get_x() + rect.get_width()/2.0, len(max_df.loc[i, 'Max Total Points Team']), 
             max_df.loc[i, 'Max Total Points Team'], ha='center', va='bottom', rotation = 'vertical')
plt.figure(figsize = (10, 8))
plt.xticks(rotation = 30)
plt.yticks([0, 5, 10, 15, 20])
g = sns.countplot(x = 'Max Total Points Team', data = max_df)
for rect in g.patches:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
