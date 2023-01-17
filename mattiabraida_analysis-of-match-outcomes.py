#Importing libraries and setting plotstyle.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

plt.style.use('seaborn-darkgrid')

sns.set(style='darkgrid', context='notebook', rc={'figure.figsize':(12,6)})
path = "../input/"

database = path + 'database.sqlite'



# 1. Establish a connection to database.

conn=sqlite3.connect(database)



# 2. Query needed data.

match_data=pd.read_sql_query('select league_id, season, date, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal from Match;', conn, parse_dates=True)

league_data=pd.read_sql_query('select id, name AS league from League;', conn)

team_data=pd.read_sql_query('select team_api_id, team_long_name from Team;', conn)
match_data.info()
league_data.info()
team_data.info()
#Tidy date column.

match_data['date']=pd.to_datetime(match_data['date'],format='%Y-%m-%d')



match_data.sort_values('date')



#Check results.

match_data.info()
#Merge match_data and league_data in order to replace league_id with league name.

match_data=pd.merge(match_data, league_data, how='inner',left_on='league_id',right_on='id')

match_data.head()
#Merge match_data and team_data in order to replace team_id with team name. 

match_data=match_data.merge(team_data, left_on='home_team_api_id',right_on='team_api_id')

match_data=match_data.merge(team_data, left_on='away_team_api_id',right_on='team_api_id',suffixes=('_a'))

match_data.head()
#Rename columns.

match_data.rename(index=str,columns={'team_long_name_':'home_team','team_long_namea':'away_team'},inplace=True)

match_data.head()
#Drop not relevant columns and ckeck final result.

match_data.drop(['league_id','home_team_api_id','away_team_api_id', 'id', 'team_api_id_','team_api_ida'], axis=1, inplace=True)

match_data.head()
#Create variable goal_diff

match_data['goal_diff']=abs(match_data['home_team_goal']-match_data['away_team_goal'])

match_data.head()
#Create variable goal_sum

match_data['goal_sum']=match_data['home_team_goal']+match_data['away_team_goal']

match_data.head()
#Create variable under_over

def under_over(goal_sum):

    if goal_sum>=3:

        value='over 2,5'

    else:

        value='under 2,5'

    return value



match_data['under_over']=match_data['goal_sum'].apply(under_over)

match_data.head()
#Create variable 1X2 

def is_1X2(home_team_goal,away_team_goal):

    if home_team_goal>away_team_goal:

        value='1'

    elif home_team_goal<away_team_goal:

        value='2'

    else:

        value='X'

    return value

match_data['1X2']=match_data.apply(lambda x: is_1X2(x['home_team_goal'],x['away_team_goal']), axis=1)

match_data.head()
#CREATING VARIABLE home1X

def home1X(home_team_goal,away_team_goal):

    if home_team_goal>=away_team_goal:

        value='yes'

    else:

        value='no'

    return value

match_data['home1X']=match_data.apply(lambda x: home1X(x['home_team_goal'],x['away_team_goal']), axis=1)
match_data['goal_sum'].describe()
#Plot the distribution of goal_sum and its evolution over the considered timeframe.

plt.figure(figsize=(12,6))

figure, axes = plt.subplots(2, 1)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

sns.distplot(match_data['goal_sum'], kde=False, ax=axes[0]).set_title('Distribution of goal scored')

sns.lineplot(x='season',y='goal_sum',data=match_data, err_style=None, ax=axes[1]).set_title('How does the variable goal_sum evolve over time?')
#Plot under_over by league

sns.countplot(y='league',hue='under_over', data=match_data)
#Create table with percentage of matches under_over for each league.

match_data.groupby(by='league')['under_over'].value_counts(normalize=True).sort_values(ascending=False)
#WHICH TEAM HAVE SCORED THE MOST GOALS?

#CREATING matche_home AND match_away and merge them into match_goal

match_home=match_data.loc[:,('date','home_team','home_team_goal')]

match_home.rename(index=str,columns={'home_team':'team','home_team_goal':'goal'},inplace=True)



match_away=match_data.loc[:,('date','away_team','away_team_goal')]

match_away.rename(index=str,columns={'away_team':'team','away_team_goal':'goal'},inplace=True)



match_goal=pd.concat([match_home, match_away], ignore_index=True)



goal_scored_by_team=match_goal.groupby(by='team')

goal_scored_by_team.mean().sort_values('goal',ascending=False).head(20).plot(kind='bar', title='Goal scored by Team')

goal_scored_by_team.mean().sort_values('goal',ascending=False).head(5)
#Plot the distribution of goal_diff and its evolution over the considered timeframe.

sns.lineplot(x='season',y='goal_diff',data=match_data, err_style=None).set_title('How does the variable goal_diff evolve over time?')
#Most equilibrated leagues.

goal_diff_by_league=match_data.groupby('league')[['league','goal_diff']]

goal_diff_by_league.mean().sort_values('goal_diff',ascending=True).plot(kind='bar', title='Most equilibrated leagues')

goal_diff_by_league.mean().sort_values('goal_diff',ascending=True).head()
#Create Series functional for plotting

serres1X2=match_data['1X2'].value_counts(normalize=True)

serhome1x=match_data['home1X'].value_counts(normalize=True)



serhome1x
#Plotting Series

f, axes = plt.subplots(1, 2)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

sns.countplot(x='1X2', data=match_data, ax=axes[0])

sns.countplot(x='home1X', data=match_data, ax=axes[1])
#Print summary statistics of home_team_goal.

match_data[['home_team_goal','away_team_goal']].describe()
#Plot the distribution of home_team_goal and its evolution over time.

f, axes = plt.subplots(2, 1)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

sns.distplot(match_data['home_team_goal'], kde=False, ax=axes[0]).set_title('Distribution of home_team_goal')

sns.distplot(match_data['away_team_goal'], kde=False, ax=axes[1]).set_title('Distribution of away_team_goal')
f, axes = plt.subplots(2, 1)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

sns.lineplot(x='season', y='goal_diff', hue='1X2', data=match_data, err_style=None, ax=axes[0]).set_title('Breakdown of goal_diff by final result')

match_data.groupby(by='season')[['home_team_goal','away_team_goal']].mean().plot(ax=axes[1], title='Difference between goals scored at home and goals scored away')