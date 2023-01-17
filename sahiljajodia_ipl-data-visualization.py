import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.plotly as py
%matplotlib inline
plt.rcParams['figure.figsize'] = [15, 5]
deliveries = pd.read_csv("../input/deliveries.csv")
matches = pd.read_csv("../input/matches.csv")
matches.head()
# len(matches.loc[matches['season'] == 2017])
all_seasons = list(set(matches.loc[:, 'season']))
#No of matches in different seasons
matches_in_seasons = [len(matches.loc[matches['season'] == i]) for i in all_seasons]
plt.bar(np.arange(len(all_seasons)), matches_in_seasons)
plt.xticks(np.arange(len(all_seasons)), all_seasons, rotation='vertical')
plt.ylabel('No. of Matches')
plt.xlabel('Seasons')
plt.title('No of matches in different seasons')
# No of matches won by different teams over the years
teams = list(set(matches.loc[:, 'team1']))
teams
matches_won = [len(matches.loc[matches['winner'] == i]) for i in teams]
plt.bar(np.arange(len(teams)), matches_won)
plt.xticks(np.arange(len(teams)), teams, rotation='vertical')
plt.ylabel('No. of Matches won')
plt.xlabel('Teams')
plt.title('No of matches won by different teams')
#Total matches played by different teams
matches_played = [len(matches.loc[matches['team1'] == i] + matches.loc[matches['team2'] == i]) for i in teams]
plt.bar(np.arange(len(teams)), matches_played)
plt.xticks(np.arange(len(teams)), teams, rotation='vertical')
plt.ylabel('No. of Matches played')
plt.xlabel('Teams')
plt.title('No of matches played by different teams')
#win/loss ratio
matches_lost = [matches_played[i] - matches_won[i] for i in range(len(teams))]
win_ratio = [(matches_won[i]/matches_played[i])*100 for i in range(len(teams))]
plt.bar(np.arange(len(teams)), win_ratio)
plt.xticks(np.arange(len(teams)), teams, rotation='vertical')
plt.ylabel('Win ratio')
plt.xlabel('Teams')
plt.title('Win ratio of different teams')
#Does winning tosses win you matches?
tosses_winning_matches_by_teams = [len(matches.loc[(matches['toss_winner'] == i) & (matches['winner'] == i)]) for i in teams]
plt.bar(np.arange(len(teams)), tosses_winning_matches_by_teams)
plt.xticks(np.arange(len(teams)), teams, rotation='vertical')
plt.ylabel('Tosses and matches both won')
plt.xlabel('Teams')
plt.title('Teams winning matches after winning tosses')
#Tosses winning matches won by teams different seasons
all_seasons = list(set(matches.loc[:, 'season']))

# tosses_winning_matches_in_seasons = []
# for i in all_seasons:
#     tosses_winning_matches_in_seasons.append(len(matches.loc[(matches['toss_winner'] == matches['winner']) & (matches['season'] == i)]))
tosses_winning_matches_in_seasons = [len(matches.loc[(matches['toss_winner'] == matches['winner']) & (matches['season'] == i)]) for i in all_seasons]
matches_in_seasons = [len(matches.loc[matches['season'] == i]) for i in all_seasons]

fig, ax = plt.subplots()
index = np.arange(len(all_seasons))
ax.bar(index, matches_in_seasons, width=0.3, label="Total")
ax.bar(index + 0.3, tosses_winning_matches_in_seasons, width=0.3, label="Toss dependent")
ax.set_xlabel("Seasons")
ax.set_ylabel("Matches")
ax.set_title("Toss dependent matches in different seasons")
plt.xticks(index, all_seasons, rotation='vertical')
ax.legend()
plt.show()

venues = list(set(matches.loc[:, 'venue']))
matches_at_different_venues = [len(matches.loc[matches['venue'] == i]) for i in venues]
plt.bar(np.arange(len(venues)), matches_at_different_venues)
plt.xticks(np.arange(len(venues)), venues, rotation='vertical')
plt.ylabel('No. of Matches')
plt.xlabel('Venues')
plt.title('No. of matches at different venues')
cities = list(set(matches.loc[:, 'city']))[1:]
matches_at_different_cities = [len(matches.loc[matches['city'] == i]) for i in cities]
plt.bar(np.arange(len(cities)), matches_at_different_cities)
plt.xticks(np.arange(len(cities)), cities, rotation='vertical')
plt.ylabel('No. of Matches')
plt.xlabel('Cities')
plt.title('No. of matches at different Cities')
teams_batting_first_in_seasons = [len(matches.loc[(matches['toss_decision'] == 'bat') & (matches['season'] == i)]) for i in all_seasons]
teams_fielding_first_in_seasons = [len(matches.loc[(matches['toss_decision'] == 'field') & (matches['season'] == i)]) for i in all_seasons]
fig, ax = plt.subplots()
index = np.arange(len(all_seasons))
ax.bar(index, teams_batting_first_in_seasons, width=0.3, label="Batting First")
ax.bar(index + 0.3, teams_fielding_first_in_seasons, width=0.3, label="Fielding First")
ax.set_xlabel("Seasons")
ax.set_ylabel("Matches")
ax.set_title("Toss decisions in different seasons")
plt.xticks(index, all_seasons, rotation='vertical')
ax.legend()
plt.show()
# result_types = list(set(matches.loc[:, 'result']))
tie_result_in_seasons = [len(matches.loc[(matches['result'] == 'tie') & (matches['season'] == i)]) for i in all_seasons]
normal_result_in_seasons = [len(matches.loc[(matches['result'] == 'normal') & (matches['season'] == i)]) for i in all_seasons]
no_result_in_seasons = [len(matches.loc[(matches['result'] == 'no result') & (matches['season'] == i)]) for i in all_seasons]

fig, ax = plt.subplots()
index = np.arange(len(all_seasons))
ax.bar(index - 0.2, no_result_in_seasons, width=0.2, label="No result")
ax.bar(index, tie_result_in_seasons, width=0.2, label="Tie")
ax.bar(index + 0.2, normal_result_in_seasons, width=0.2, label="Normal")
ax.set_xlabel("Seasons")
ax.set_ylabel("Matches")
ax.set_title("Types of results")
plt.xticks(index, all_seasons, rotation='vertical')
ax.legend()
plt.show()

dl_applied_matches_in_seasons = [len(matches.loc[(matches['dl_applied'] == 1) & (matches['season'] == i)]) for i in all_seasons]
plt.bar(np.arange(len(all_seasons)), dl_applied_matches_in_seasons)
plt.xticks(np.arange(len(all_seasons)), all_seasons, rotation='vertical')
plt.ylabel('No. of Matches')
plt.xlabel('Seasons')
plt.title('No. of matches at in which DL was applied')
