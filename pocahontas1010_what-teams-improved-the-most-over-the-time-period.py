# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
path = "../input/"
print(os.listdir("../input/belgium-2013-2014"))
# path to the database
database = path + 'soccer/database.sqlite'

# connection to database
connection = sqlite3.connect(database)

# read sql file
query = open('../input/belgium-2013-2014/matches.sql', 'r')
df_matches = pd.read_sql_query(query.read(), connection)

# save the the data into a csv file
df_matches.to_csv('matches.csv', index=False)

df_matches.head()
# number of samples and columns
df_matches.shape
# check for duplicates
sum(df_matches.duplicated())
# check the datatypes
df_matches.info()
# check for missing values
df_matches.isnull().sum()
# non-null unique values
df_matches.nunique()
# describe the dataset
df_matches.describe()
# games per country, per match_season, per team
game_played = df_matches.groupby(['country', 'match_season', 'team']).sum().game_played
game_played
# view the teams where the number of games they played is higher than 75%
game_played[game_played>38]
# see the data with incorrect values
df_matches.query('(team == "Polonia Bytom" and (match_season == "2010/2011" or match_season == "2008/2009")) or (team == "Widzew Łódź" and match_season == "2011/2012")')
# change the number of game played
df_matches.loc[949, 'game_played'] = 30
df_matches.loc[964, 'game_played'] = 30
df_matches.loc[996, 'game_played'] = 30

# change the number of points
df_matches.loc[949, 'points'] = 39
df_matches.loc[964, 'points'] = 27
df_matches.loc[996, 'points'] = 35

# change the number of games they won
df_matches.loc[949, 'won'] = 9
df_matches.loc[964, 'won'] = 6
df_matches.loc[996, 'won'] = 10

# change the number of draw games 
df_matches.loc[949, 'draw'] = 12
df_matches.loc[964, 'draw'] = 9
df_matches.loc[996, 'draw'] = 5

# change the number of games they lost
df_matches.loc[949, 'lost'] = 9
df_matches.loc[964, 'lost'] = 15
df_matches.loc[996, 'lost'] = 15

# change the number of goals scored 
df_matches.loc[949, 'goals_scored'] = 25
df_matches.loc[964, 'goals_scored'] = 29
df_matches.loc[996, 'goals_scored'] = 30

# change the number of goals conceded
df_matches.loc[949, 'goals_conceded'] = 26
df_matches.loc[964, 'goals_conceded'] = 45
df_matches.loc[996, 'goals_conceded'] = 46

# change the number of goals difference
df_matches.loc[949, 'goals_difference'] = -1
df_matches.loc[964, 'goals_difference'] = -16
df_matches.loc[996, 'goals_difference'] = -16

# chack again to see the right values
df_matches.query('(team == "Polonia Bytom" and (match_season == "2010/2011" or match_season == "2008/2009")) or (team == "Widzew Łódź" and match_season == "2011/2012")')
# save the changes datasets to csv
df_matches.to_csv('matches.csv', index=False)

# load dataset
df_matches = pd.read_csv('matches.csv')
df_matches.head()
# games per country, per match_season, per team
game_played = df_matches.groupby(['country', 'match_season', 'team']).sum().game_played
# check if  the number of games played is consistent
game_played[game_played>38]
pd.set_option('display.max_rows', 5400)
match_teams = df_matches.groupby(['country', 'match_season'], as_index=False).count().loc[:,['country', 'match_season', 'team']]
match_teams.head()
# check if the number of teams in match season is consistent
match_teams[match_teams['team'] < 10]
# drop the rows with Belgium data for 2013/2014 match_season
df_matches.drop(df_matches.query('country == "Belgium" and match_season == "2013/2014"').index, inplace=True)
df_matches.shape
df_matches.head(1)
# # upload correct dataset for belgium
df_belgium_2013_2014 = pd.read_csv('../input/belgium-2013-2014/belgium_2013_2014.csv')
df_belgium_2013_2014.head()
#print(os.listdir("../input"))
#path = "../input/"
#print(os.listdir("../input/belgium-2013-2014"))

# combine the two dataframes 
df = pd.concat([df_matches, df_belgium_2013_2014])

# save the datasets to csv
df.to_csv('matches.csv', index=False)
df = pd.read_csv('matches.csv')

# view the first rows of the dataset
df.head()
# cheack if the data is appended
match_teams = df.groupby(['country', 'match_season'], as_index=False).count().loc[:,['country', 'match_season', 'team']]
match_teams[match_teams['team'] < 10].count()
df.shape
df.describe()
# how many points each team got per country, per season match
df.groupby(['country', 'league', 'match_season']).count().head()
pd.plotting.scatter_matrix(df, figsize=(15,15));
# plot the points histogram

plt.hist(df['points'])
plt.title('Distribution of Points')
plt.xlabel('Points')
plt.ylabel('Frequency')

plt.show;
# calculate the correlation coefficient
corr = np.round(df['points'].corr(df['won']), decimals=2)

# plot the scatter
plt.scatter(df['points'], df['won'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Won Matches and Points')
plt.xlabel('Points')
plt.ylabel('Won')

plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
df.plot(x='points', y='goals_scored', kind='scatter');

# calculate the correlation coefficient
corr = np.round(df['points'].corr(df['goals_scored']), decimals=2)

# plot the scatter
plt.scatter(df['points'], df['goals_scored'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Gained Points and Scored Goals')
plt.xlabel('Points')
plt.ylabel('Scored Goals')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# calculate the correlation coefficient
corr = np.round(df['points'].corr(df['lost']), decimals=2)

# plot the scatter
plt.scatter(df['points'], df['lost'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Gained Points and Lost Matches')
plt.xlabel('Points')
plt.ylabel('Lost Matches')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# calculate the correlation coefficient
corr = np.round(df['points'].corr(df['goals_difference']), decimals=2)

# plot the scatter
plt.scatter(df['points'], df['goals_difference'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Gained Points and Goals Difference')
plt.xlabel('Points')
plt.ylabel('Goals Difference')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# plot the won matches histogram
plt.hist(df['won'])
plt.title('Distribution of Won Matches')
plt.xlabel('Won')
plt.ylabel('Frequency')

plt.show;
# calculate the correlation coefficient
corr = np.round(df['won'].corr(df['goals_scored']), decimals=2)

# plot the scatter
plt.scatter(df['won'], df['goals_scored'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Won Matches and Scored Goals')
plt.xlabel('Won')
plt.ylabel('Scored Goals')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# calculate the correlation coefficient
corr = np.round(df['won'].corr(df['goals_conceded']), decimals=2)

# plot the scatter
plt.scatter(df['won'], df['goals_conceded'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Won Matches and Goals Conceded')
plt.xlabel('Won')
plt.ylabel('Goals Conceded')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# calculate the correlation coefficient
corr = np.round(df['won'].corr(df['goals_difference']), decimals=2)

# plot the scatter
plt.scatter(df['won'], df['goals_difference'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Won Matches and Goals Difference')
plt.xlabel('Won')
plt.ylabel('Goals Difference')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# plot the draw matches histogram
plt.hist(df['draw'])
plt.title('Distribution of Draw Matches')
plt.xlabel('Draw')
plt.ylabel('Frequency')

plt.show;
# plot the draw matches histogram
plt.hist(df['lost'])
plt.title('Distribution of Lost Matches')
plt.xlabel('Lost')
plt.ylabel('Frequency')

plt.show;
# calculate the correlation coefficient
corr = np.round(df['lost'].corr(df['goals_difference']), decimals=2)

# plot the scatter
plt.scatter(df['lost'], df['goals_difference'], label="Correlation Coefficient {}".format(corr));
plt.title('Scatterplot of Lost Matches and Goals Difference')
plt.xlabel('Lost')
plt.ylabel('Goals Difference')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.show();
# plot the goals scored histogram
plt.hist(df['goals_scored'])
plt.title('Distribution of Goals Scored')
plt.xlabel('Goals Scored')
plt.ylabel('Frequency')

plt.show;
# plot the goals scored histogram
plt.hist(df['goals_conceded'])
plt.title('Distribution of Goals Conceded')
plt.xlabel('Goals Conceded')
plt.ylabel('Frequency')

plt.show;
# plot the goals scored histogram
plt.hist(df['goals_difference'])
plt.title('Distribution of Goals Difference')
plt.xlabel('Goals Difference')
plt.ylabel('Frequency')

plt.show;
# view the first row of the dataset
df.head(1)
countries = df.country.unique()
leagues = df.league.unique()
match_seasons = df.match_season.unique()
countries, leagues, match_seasons
df.groupby(['country', 'league', 'match_season'], as_index=False).max().head()
# group the data by the number of points goals_difference  
df_points = df.groupby(['country', 'league', 'match_season', 'team'], as_index=False)['points', 'goals_difference'].sum()
# df belgium 2008/2009
df_belgium = df_points[df_points['country'] == 'Belgium']
def filter_country(country):
    df_country = df_points[df_points['country'] == country]
    # sorting the data
    df_country = df_country.sort_values(by=['match_season', 'points', 'goals_difference'], ascending=False)
    return df_country
# Belgium filtered data
df_belgium = filter_country('Belgium')

# save the datasets to csv
df_belgium.to_csv('df_belgium.csv', index=False)

# load dataset 
df_belgium = pd.read_csv('df_belgium.csv')

df_belgium.head()
def country_details(df_country):
    country = df_country['country'].unique()
    league = df_country['league'].unique()
    match_season = df_country['match_season'].unique()
    return country, league, match_season
country, league, match_season = country_details(df_belgium)
country, league, match_season
def filter_country_plot(df_country, season):
    country, league, match_season = country_details(df_country)
    
    # initialize values 
    team = []
    points = []
    goals_df = []
    
    for index, row in df_country.iterrows():
            if row['match_season'] == season:
                # append values
                team.append(row['team'])
                points.append(row['points'])
                goals_df.append( row['goals_difference'])
    
    # convert list to NumPy array            
    teams = np.array(team)
    points = np.array(points)
    goals_df = np.array(goals_df)   

    # get the top 5 values
    top_5_teams = teams[:5]
    top_5_points = points[:5]
    top_5_goals_df = goals_df[:5]

    # get the last 3 values
    last_3_position = len(teams) - 3
    last_3_teams = teams[last_3_position :]
    last_3_points = points[last_3_position :]
    last_3_goals_df = goals_df[last_3_position :]
    
    return top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df
top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df = filter_country_plot(df_belgium, match_season[0])
top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df
def plot_barh(teams, points, goals_df, season):
    width = 0.35
    ind = np.arange(len(teams))
    locations = ind + width / 2 # ytick locations
    labels = teams # ytick labels
    
    if len(teams) == 5:
        plt_title = 'Top Five Winners'
    else:
        plt_title = 'Last Three Teams'
    
    heights_points = points
    heights_gd = goals_df
    winners_points = plt.barh(ind, heights_points, width, alpha=.7, label='Points')
    winner_gd = plt.barh(ind + width, heights_gd, width, alpha=.7, label='Goals Difference')

    # title and labels
    plt.title('{} in {}, {} Match Season, {} by Points and Goals Difference'.format(plt_title, league[0], season, country[0]))
    plt.xlabel('Points/Goal differences')
    plt.ylabel('Team')
    plt.yticks(locations, labels)

    #legend
    plt.legend()

    plt.show();
plot_barh(top_5_teams, top_5_points, top_5_goals_df, match_season[0])
plot_barh(last_3_teams, last_3_points, last_3_goals_df, match_season[0])
for season in match_season:
    top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df = filter_country_plot(df_belgium, season)
    plot_barh(top_5_teams, top_5_points, top_5_goals_df, season)
    plot_barh(last_3_teams, last_3_points, last_3_goals_df, season)
# Italy filtered data
df_italy = filter_country('Italy')

# save the datasets to csv
df_italy.to_csv('df_italy.csv', index=False)

# load dataset 
df_italy = pd.read_csv('df_italy.csv')

df_italy.head()
country, league, match_season = country_details(df_italy)
country, league, match_season
top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df = filter_country_plot(df_italy, match_season[0])
plot_barh(top_5_teams, top_5_points, top_5_goals_df, match_season[0])
plot_barh(last_3_teams, last_3_points, last_3_goals_df, match_season[0])
for season in match_season:
    top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df = filter_country_plot(df_italy, season)
    plot_barh(top_5_teams, top_5_points, top_5_goals_df, season)
    plot_barh(last_3_teams, last_3_points, last_3_goals_df, season)
df_italy.head()
def top_5_teams(df_country):
    top_5_all= []
    for season in match_season:
        top_5_teams, top_5_points, top_5_goals_df, last_3_teams, last_3_points, last_3_goals_df = filter_country_plot(df_country, season)
        top_5_all.append(top_5_teams)
    top_5_all = np.unique(top_5_all)
    return top_5_all

top_5 = top_5_teams(df_italy)
top_5
def points_plot_chart(df_country, team):
    
    # Filter database by first team
    df_top_5 = df_country[df_country['team'] == team]
    df_top_5 = df_top_5.sort_values(by=['match_season'], ascending=True)
    
    # Get Only `match_season` and `team`
    df_top_5.loc[:, ['match_season', 'points']]
    
    # plot the data
    labels = df_top_5.loc[:, 'match_season']
    heights = df_top_5.loc[:, 'points']
    plt.plot(labels, heights)

    # titles
    plt.ylabel("Total Points")
    plt.xlabel("Match Season (Years)")
    plt.xticks(labels, rotation=30)
    plt.title('{} Points by Match Season'.format(team))

    plt.show()
    
#points_plot_chart(df_italy, 'Fiorentina')
for team in top_5:
    points_plot_chart(df_italy, team)
df_top_5 = df_italy[df_italy['team'].isin(top_5)]
top_5
df_top_5 = df_top_5.groupby(['match_season', 'team'], as_index=False).sum().sort_values(by=['team', 'match_season'])
df_top_5
labels = df_top_5[df_top_5['team'] == 'Fiorentina'].loc[:, 'match_season']
plt.plot(labels, df_top_5[df_top_5['team'] == 'Fiorentina'].loc[:, 'points'], label='Fiorentina')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Genoa'].loc[:, 'points'], label='Genoa')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Inter'].loc[:, 'points'], label='Inter')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Juventus'].loc[:, 'points'], label='Juventus')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Lazio'].loc[:, 'points'], label='Lazio')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Milan'].loc[:, 'points'], label='Milan')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Napoli'].loc[:, 'points'], label='Napoli')
plt.plot(df_top_5[df_top_5['team'] == 'Palermo'].loc[:, 'match_season'], df_top_5[df_top_5['team'] == 'Palermo'].loc[:, 'points'], label='Palermo')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Roma'].loc[:, 'points'], label='Roma')
plt.plot(df_top_5[df_top_5['team'] == 'Sampdoria'].loc[:, 'match_season'], df_top_5[df_top_5['team'] == 'Sampdoria'].loc[:, 'points'], label='Sampdoria')
plt.plot(labels, df_top_5[df_top_5['team'] == 'Udinese'].loc[:, 'points'], label='Udinese')
plt.xticks(labels, rotation=30)

#plt.figure(figsize=(8,3))
#plt.get_current_fig_manager().resize(400, 200)
plt.title('Points per Match Season by Ever Ranked Top Five Teams in Italy Serie A')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show();
