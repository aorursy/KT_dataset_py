import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from scipy.stats import spearmanr

%matplotlib inline
seasons = pd.read_csv('../input/Seasons.csv', index_col=0)

teams = pd.read_csv('../input/Teams.csv', index_col=0)

compact_results_regular = pd.read_csv('../input/RegularSeasonCompactResults.csv')

compact_results_tourney = pd.read_csv('../input/TourneyCompactResults.csv')
teams_won_most_times = teams.join(pd.DataFrame(compact_results_regular['Wteam'].value_counts()[:5]), how='right')

teams_won_most_times.columns = ['Team_Name', 'Wins_number']

teams_won_most_times
plt.hist(compact_results_regular['Wteam'].value_counts(), bins=20)

plt.ylabel('number of teams')

plt.xlabel('number of victories')
teams_lost_most_times = teams.join(pd.DataFrame(compact_results_regular['Lteam'].value_counts()[:5]), how='right')

teams_lost_most_times.columns = ['Team_Name', 'Losses_number']

teams_lost_most_times
plt.hist(compact_results_regular['Lteam'].value_counts(), bins=20)

plt.ylabel('number of teams')

plt.xlabel('number of defeates')
number_teams_wins = pd.DataFrame(compact_results_regular['Wteam'].value_counts())

number_teams_losses = pd.DataFrame(compact_results_regular['Lteam'].value_counts())

number_teams_matches = pd.DataFrame(number_teams_wins['Wteam'] + number_teams_losses['Lteam'])

number_teams_matches.columns = ['Matches_number']
plt.hist(number_teams_matches['Matches_number'], bins=20)

plt.ylabel('number of teams')

plt.xlabel('number of matches')
statictics_table = number_teams_matches.join([number_teams_wins, number_teams_losses])

print(spearmanr(statictics_table['Matches_number'], statictics_table['Wteam']))

print(spearmanr(statictics_table['Matches_number'], statictics_table['Lteam']))
plt.plot(statictics_table['Matches_number'], statictics_table['Wteam'], 'o')

plt.xlabel('number of matches')

plt.ylabel('number of victories')
plt.plot(statictics_table['Matches_number'], statictics_table['Lteam'], 'o')

plt.xlabel('number of matches')

plt.ylabel('number of defeates')
teams_regular_results = dict()



for team in teams.index:

    teams_regular_results[team] = list()

    

for season in seasons.index:

    one_season_matches = compact_results_regular[compact_results_regular['Season'] == season]

    one_season_wins_number = pd.DataFrame(one_season_matches['Wteam'].value_counts())

    one_season_wins_number.columns = ['Wins_number']

    one_season_teams = set(one_season_matches['Lteam'].unique()).union(set(one_season_matches['Wteam'].unique()))

    for team in teams_regular_results.keys():

        if team in one_season_teams:

            if team in one_season_wins_number.index:

                teams_regular_results[team].append(one_season_wins_number.loc[team]['Wins_number'])

            else:

                teams_regular_results[team].append(0)

        else:

            teams_regular_results[team].append(-5)   
team_name = 1108

plt.plot(teams_regular_results[team_name], '-o')

plt.plot([-5] * (len(seasons.index) + 1), '--')

plt.text(-10, -5, "no participate")

plt.ylim(-6, 23)

plt.ylabel('number of wins')

plt.xlabel('season')

plt.title('Team ' + teams.loc[team_name]['Team_Name'])

plt.xticks(range(len(seasons))[::5], seasons.index[::5])

plt.show()
team_name = 1242

plt.plot(teams_regular_results[team_name], '-o')

plt.plot([-5] * (len(seasons.index) + 1), '--')

plt.text(-10, -5, "no participate")

plt.ylim(-6, 40)

plt.ylabel('number of wins')

plt.xlabel('season')

plt.title('Team ' + teams.loc[team_name]['Team_Name'])

plt.xticks(range(len(seasons))[::5], seasons.index[::5])

plt.show()
team_name = 1392

plt.plot(teams_regular_results[team_name], '-o')

plt.plot([-5] * (len(seasons.index) + 1), '--')

plt.text(-10, -5, "no participate")

plt.ylim(-6, 28)

plt.ylabel('number of wins')

plt.xlabel('season')

plt.title('Team ' + teams.loc[team_name]['Team_Name'])

plt.xticks(range(len(seasons))[::5], seasons.index[::5])

plt.show()
teams_won_most_times = teams.join(pd.DataFrame(compact_results_tourney['Wteam'].value_counts()[:5]), how='right')

teams_won_most_times.columns = ['Team_Name', 'Wins_number']

teams_won_most_times
teams_lost_most_times = teams.join(pd.DataFrame(compact_results_tourney['Lteam'].value_counts()[:5]), how='right')

teams_lost_most_times.columns = ['Team_Name', 'Losses_number']

teams_lost_most_times
plt.hist(compact_results_tourney['Wteam'].value_counts(), bins=20)

plt.ylabel('number of teams')

plt.xlabel('number of victories')
plt.hist(compact_results_tourney['Lteam'].value_counts(), bins=20)

plt.ylabel('number of teams')

plt.xlabel('number of defeates')
number_teams_wins = pd.DataFrame(compact_results_tourney['Wteam'].value_counts())

number_teams_losses = pd.DataFrame(compact_results_tourney['Lteam'].value_counts())

number_teams_matches = pd.concat([number_teams_wins, number_teams_losses], axis=1)

number_teams_matches.fillna(0, inplace=True)



number_teams_matches['Matches_number'] = number_teams_matches['Wteam'] + number_teams_matches['Lteam']

plt.hist(number_teams_matches['Matches_number'], bins=20)

plt.ylabel('number of teams')

plt.xlabel('number of matches')
teams_tourney_results = dict()



for team in teams.index:

    teams_tourney_results[team] = list()

 

for season in seasons.index:

    one_season_matches = compact_results_tourney[compact_results_tourney['Season'] == season]

    one_season_wins_number = pd.DataFrame(one_season_matches['Wteam'].value_counts())

    one_season_wins_number.columns = ['Wins_number']

    one_season_teams = set(one_season_matches['Lteam'].unique()).union(set(one_season_matches['Wteam'].unique()))

    for team in teams_tourney_results.keys():

        if team in one_season_teams:

            if team in one_season_wins_number.index:

                teams_tourney_results[team].append(one_season_wins_number.loc[team]['Wins_number'])

            else:

                teams_tourney_results[team].append(0)

        else:

            teams_tourney_results[team].append(-5)
team_name = 1181

plt.plot(teams_tourney_results[team_name], '-o')

plt.plot([-5] * (len(seasons.index) + 1), '--')

plt.text(-10, -5, "no participate")

plt.ylim(-6, 10)

plt.ylabel('number of wins')

plt.xlabel('season')

plt.title('Team ' + teams.loc[team_name]['Team_Name'])

plt.xticks(range(len(seasons))[::5], seasons.index[::5])

plt.show()
team_name = 1380

plt.plot(teams_tourney_results[team_name], '-o')

plt.plot([-5] * (len(seasons.index) + 1), '--')

plt.text(-10, -5, "no participate")

plt.ylim(-6, 10)

plt.ylabel('number of wins')

plt.xlabel('season')

plt.title('Team ' + teams.loc[team_name]['Team_Name'])

plt.xticks(range(len(seasons))[::5], seasons.index[::5])

plt.show()
results_correlation_coef = list()



for team in teams_regular_results.keys():

    if np.any(np.unique(teams_tourney_results[team]) != np.array([-5])): 

        results_correlation_coef.append(spearmanr(teams_regular_results[team], teams_tourney_results[team])[0])
np.mean(results_correlation_coef)
plt.hist(results_correlation_coef)

plt.xlabel('spearman correlation')

plt.ylabel('number of teams')