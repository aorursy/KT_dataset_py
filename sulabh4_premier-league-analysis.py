import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

%matplotlib inline
# # # Given
start_season_year = 1992
end_season_year = 2018
total_seasons = 26   # # (1992-2018)
total_teams = 20
relegation_count = 3
relegation_threshold = 18
column_headers = ['Teams', 'M', 'W', 'D', 'L', 'Goals', 'Diff', 'Pts']
data_dictionary = {}
for i in range(total_seasons):
    season_date = i + start_season_year
    data_dictionary[season_date] = pd.read_csv('../input/https___www.worldfootball.net_ ({0}).csv'.format(i), header=None, index_col=0)
    
    individual_dataframe = data_dictionary[season_date]
    
    # # Remove unknown columns
    del individual_dataframe[9]
    del individual_dataframe[10]
    
    # # Remove unknown index
    individual_dataframe.drop(individual_dataframe.index[:2], inplace=True)
    
    # # Good Columns Names
    individual_dataframe.columns = column_headers
    
    individual_dataframe = individual_dataframe.rename_axis(None, inplace=True)
data_dictionary[2003]
# # # Creating a Winner's dictionary with clubs and their respective title winning years
winners_dict = {}
winners_hist = {}
for i in range(total_seasons):
    season_date = i + start_season_year
    winner_club = data_dictionary[season_date].loc['1'][0]
    winners_dict[season_date] = winner_club
    
    # # For Winners Histogram
    if winner_club in winners_hist:
        winners_hist[winner_club] += 1
    else:
        winners_hist[winner_club] = 1
winners_hist
winners_dict
width = 0.5
figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')

y_ticks = np.arange(0, 15)
plt.yticks(y_ticks)

plt.bar(winners_hist.keys(), winners_hist.values(), width, color='green')
# # Getting the difference in points season
difference_hist = {}
difference_points = []
for i in range(total_seasons):
    season_date = i + start_season_year

    winner_point = data_dictionary[season_date].loc['1'][7]
    loser_point = data_dictionary[season_date].loc['20'][7]
    
    difference_points.append(int(winner_point) - int(loser_point))
    difference_hist[season_date] = (int(winner_point), int(loser_point))
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
x = list(difference_hist.values())
data = list(difference_hist.keys())

plt.boxplot(x=x, data=data, labels=data, showmeans=True)
plt.show()
best_place_dict = {}
best_place_date_dict = {}
for i in range(total_seasons):
    season_date = i + start_season_year
    for j in range(total_teams):
        team_name = data_dictionary[season_date].loc[str(j+1)][0]
        # # Check whether the team is already in the dictionary
        if team_name in best_place_dict:
            # # Compare the best place from the old place
            if best_place_dict[team_name] > j+1:
                best_place_dict[team_name] = j+1
        else:
            best_place_dict[team_name] = j+1
            best_place_date_dict[team_name] = season_date
figure(num=None, figsize=(15, 25), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Best Place Finish')
plt.ylabel('Clubs')

x_ticks = np.arange(1, 21)

plt.grid(axis='both')
plt.xticks(x_ticks)
plt.scatter(x=best_place_dict.values(), y=best_place_dict.keys())
for i in range(total_seasons):
    season_date = i + start_season_year
    for j in range(relegation_count):
        team_name = data_dictionary[season_date].loc[str(relegation_threshold + j)][0]
        if team_name in best_place_date_dict:
            del best_place_date_dict[team_name]
figure(num=None, figsize=(26, 15), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Inception Year')
plt.ylabel('Clubs')

x_tick = np.arange(1992, 2018, 1)

plt.grid(axis='both')
plt.xticks(x_tick)
plt.scatter(x=best_place_date_dict.values(), y=best_place_date_dict.keys())
total_wins_dict = {}
for i in range(total_seasons):
    season_date = i + start_season_year
    for j in range(total_teams):
        team_name = data_dictionary[season_date].loc[str(j+1)][0]
        wins = data_dictionary[season_date].loc[str(j+1)][2]
        if team_name in total_wins_dict:
            total_wins_dict[team_name] += int(wins)
        else:           
            total_wins_dict[team_name] = int(wins)
figure(num=None, figsize=(15, 25), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Number of Wins')
plt.ylabel('Clubs')

x_ticks = np.arange(5, 630, 30)

plt.grid(axis='both')
plt.xticks(x_ticks)
plt.scatter(x=total_wins_dict.values(), y=total_wins_dict.keys())
