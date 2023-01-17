%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
matches = pd.read_csv('../input/WorldCupMatches.csv').drop_duplicates()
cups = pd.read_csv('../input/WorldCups.csv')
initials = pd.unique(pd.concat([matches['Home Team Initials'], matches['Away Team Initials']]).dropna())
countries = dict(zip(matches['Home Team Initials'], matches['Home Team Name']))
del(countries[np.nan])
cups
# calculate stats for each team
stats = pd.DataFrame()
for i in countries.keys():
    home_games = matches[matches['Home Team Initials'] == i]
    away_games = matches[matches['Away Team Initials'] == i]
    all_games = pd.concat([home_games, away_games])
    won = pd.concat([home_games[matches['Home Team Goals'] > matches['Away Team Goals']], away_games[matches['Home Team Goals'] < matches['Away Team Goals']]])
    draws = all_games[matches['Home Team Goals'] == matches['Away Team Goals']]
    total_goals = int(home_games['Home Team Goals'].sum() + away_games['Away Team Goals'].sum())
    total_goals_against = int(home_games['Away Team Goals'].sum() + away_games['Home Team Goals'].sum())
    finals = all_games[all_games['Stage'] == 'Final']
    semis = all_games[all_games['Stage'] == 'Semi-finals']
    champions = len(cups[cups['Winner'] == countries[i]])
    stats = stats.append(pd.DataFrame({'Total Games': [len(all_games)], 
                                       'Wins': [len(won)], 
                                       'Draws': [len(draws)], 
                                       'Loses': [len(all_games) - len(won) - len(draws)],
                                       'Goals': [total_goals],
                                       'Goals Against': [total_goals_against],
                                       'Champions' : [champions],
                                       'Finals' : [len(finals)],
                                       'Semi-Finals' : [len(semis)]}, index=[i]))
for i in ['Total Games', 'Wins', 'Goals', 'Goals Against']:
    stats.sort_values([i], ascending=False).head(10).plot.bar(y=i, figsize=(30,10))
stats
countries_per_ncups = dict()
max_cups = max(stats['Champions'])
for i in range(max_cups+1):
    countries_per_ncups[i] = ','.join(stats[stats['Champions'] == i].index.values)
countries_per_ncups
for i in ['GoalsScored', 'QualifiedTeams', 'MatchesPlayed']:
    cups.plot(x='Year', y=i, figsize=(20,4))