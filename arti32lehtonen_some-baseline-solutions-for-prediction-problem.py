import sqlite3 

import pandas as pd 

import numpy as np



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



import matplotlib.pyplot as plt

%matplotlib inline
with sqlite3.connect('../input/database.sqlite') as con:

    countries = pd.read_sql_query("SELECT * from Country", con, index_col='id')

    matches = pd.read_sql_query("SELECT * from Match", con, index_col='id')

    leagues = pd.read_sql_query("SELECT * from League", con,  index_col='id')

    teams = pd.read_sql_query("SELECT * from Team", con,  index_col='id')
leagues = leagues.drop(['country_id'], axis=1)

matches = matches.drop(['country_id'], axis=1)
matches = matches[matches.columns[:9]]

matches['result'] = np.sign(matches.home_team_goal - matches.away_team_goal)

matches = matches.drop(['date', 'home_team_goal', 'away_team_goal'], axis=1)

matches.stage = matches.stage.apply(lambda x: int(x))
matches.head()
def get_points_for_all_matches(matches):

    all_matches_scores = pd.DataFrame()



    for league in matches.league_id.unique():

        matches_of_one_league = matches[matches['league_id'] == league]

        all_seasons_scores = get_points_for_one_league(matches_of_one_league)

        all_matches_scores = pd.concat([all_matches_scores, all_seasons_scores])

    return all_matches_scores





def get_points_for_one_league(matches_of_one_league):

    all_seasons_scores = pd.DataFrame()



    for season in matches_of_one_league.season.unique():

        matches_of_one_season = matches_of_one_league[matches_of_one_league['season'] == season]

        one_season_scores = get_points_for_one_season(matches_of_one_season, season)

        all_seasons_scores = pd.concat([all_seasons_scores, one_season_scores])

    return all_seasons_scores





def get_points_for_one_season(matches_of_one_season, season):

    previous_stage_scores = None

    all_teams_of_season = set(matches_of_one_season.home_team_api_id.unique())

    

    all_stages_scores = pd.DataFrame([0] * len(all_teams_of_season), index=list(all_teams_of_season),

                                     columns=['result'])

    

    all_stages_scores.index = pd.MultiIndex.from_tuples(list(zip(list(all_teams_of_season), 

                                                           len(all_teams_of_season) * [season],

                                                           len(all_teams_of_season) * [1])))

    

    for stage in np.sort(matches_of_one_season.stage.unique()):

        matches_of_one_stage = matches_of_one_season[matches_of_one_season['stage'] == stage]        

        one_stage_scores = get_points_for_one_stage(matches_of_one_stage, previous_stage_scores,

                                                    all_teams_of_season)

        previous_stage_scores = pd.DataFrame(one_stage_scores, copy=True)



        one_stage_scores.index = pd.MultiIndex.from_tuples(list(zip(list(one_stage_scores.index), 

                                                           len(one_stage_scores.index) * [season],

                                                           len(one_stage_scores.index) * [stage + 1])))



        all_stages_scores = pd.concat([all_stages_scores, one_stage_scores])

    return all_stages_scores





def get_points_for_one_stage(matches_of_one_stage, previous_stage_scores,

                             all_teams_of_season):

    win_teams1 = matches_of_one_stage[matches_of_one_stage['result'] == 1]['home_team_api_id']

    win_teams2 = matches_of_one_stage[matches_of_one_stage['result'] == -1]['away_team_api_id']

    win_teams = pd.concat([win_teams1, win_teams2])

    win_teams = pd.DataFrame([3] * win_teams.shape[0],

                             win_teams, ['result'])



    drawn_teams1 = matches_of_one_stage[matches_of_one_stage['result'] == 0]['home_team_api_id']

    drawn_teams2 = matches_of_one_stage[matches_of_one_stage['result'] == 0]['away_team_api_id']

    drawn_teams = pd.concat([drawn_teams1, drawn_teams2])

    drawn_teams = pd.DataFrame([1] * drawn_teams.shape[0],

                              drawn_teams, ['result'])



    lose_teams1 = matches_of_one_stage[matches_of_one_stage['result'] == 1]['away_team_api_id']

    lose_teams2 = matches_of_one_stage[matches_of_one_stage['result'] == -1]['home_team_api_id']

    lose_teams = pd.concat([lose_teams1, lose_teams2])

    lose_teams = pd.DataFrame([0] * lose_teams.shape[0],

                              lose_teams, ['result'])

    

    played_teams = pd.concat([win_teams, drawn_teams, lose_teams])

    

    

    if len(all_teams_of_season) != len(played_teams):

        nonplayed_teams = list(all_teams_of_season.difference(set(played_teams.index)))

        nonplayed_teams =  pd.DataFrame([0] * len(nonplayed_teams),

                                  nonplayed_teams,

                                  ['result'])

        stage_scores = pd.concat([played_teams, nonplayed_teams])

    else:

        stage_scores = pd.concat([win_teams, drawn_teams, lose_teams])



    if previous_stage_scores is not None:

        stage_scores += previous_stage_scores

        

    return stage_scores
points = get_points_for_all_matches(matches)
matches = matches.join(points, on=['home_team_api_id', 'season', 'stage'], rsuffix='_home')

matches = matches.join(points, on=['away_team_api_id', 'season', 'stage'], rsuffix='_away')

matches.columns = (list(matches.columns[:-2]) + ['points_home', 'points_away'])
matches.head()
result_if_home_team_always_win = np.ones(matches.shape[0])

print(accuracy_score(matches.result, result_if_home_team_always_win))
temp_diff = np.array(np.int64((matches.points_home - matches.points_away)))

true_result = np.array(matches.result)

list_a = sorted(list(set(temp_diff)))

results_a = list()



for a in list_a:

    res = accuracy_score(true_result, np.sign(temp_diff - a))

    results_a.append(res)
plt.plot(list_a, results_a, '-', linewidth=2)

plt.plot(list_a, [accuracy_score(true_result, result_if_home_team_always_win)] * len(list_a), '--', linewidth=2)

plt.xlim(list_a[0], list_a[-1])

plt.xticks(np.arange(-60, 65, 10))

plt.ylim(0.27, 0.5)

plt.grid()

plt.title('Accuracy dependence on a')

plt.xlabel('a')

plt.ylabel('accuracy')
print(np.max(results_a), '; a =', list_a[np.argmax(results_a)])
pred = np.sign(temp_diff + 8)
confusion_matrix(true_result, pred)
temp_diff = np.array(np.int64((matches.points_home - matches.points_away)))

true_result = np.array(matches.result)



list_a = sorted(list(set(temp_diff)))



results_a2 = list()



for a in (list_a):

    temp_pred = np.zeros(temp_diff.shape)

    temp_pred[temp_diff >= a] = 1

    temp_pred[temp_diff < a] = -1

    res = accuracy_score(true_result, temp_pred)

    results_a2.append(res)
plt.plot(list_a, results_a2, '-', linewidth=2)

plt.plot(list_a, [np.max(results_a)] * len(list_a), '--', linewidth=2)

plt.xlim(-50, 40)

plt.xticks(np.arange(-60, 65, 10))

plt.ylim(0.3, 0.5)

plt.grid()

plt.title('Accuracy dependence on a')

plt.xlabel('a')

plt.ylabel('accuracy')
np.max(results_a2)
d = dict()

for league in matches.league_id.unique():

    d[league] = list()
for league in matches.league_id.unique():

    matches_of_league = matches[matches['league_id'] == league]

    

    temp_diff = np.array(np.int64((matches_of_league.points_home - matches_of_league.points_away)))

    temp_pred = np.array(np.int64(np.sign(matches_of_league.points_home - matches_of_league.points_away)))

    result_if_home_team_always_win = np.ones(matches_of_league.shape[0])

    true_result = np.array(matches_of_league.result)

    

    d[league].append(accuracy_score(true_result, result_if_home_team_always_win))
for league in matches.league_id.unique():

    matches_of_league = matches[matches['league_id'] == league]

    

    temp_diff = np.array(np.int64((matches_of_league.points_home - matches_of_league.points_away)))

    temp_pred = np.array(np.int64(np.sign(matches_of_league.points_home - matches_of_league.points_away)))

    result_if_home_team_always_win = np.ones(matches_of_league.shape[0])

    true_result = np.array(matches_of_league.result)

    

    list_a = sorted(list(set(temp_diff)))

    results_a = list()

    for a in list_a:

        temp_pred = np.zeros(temp_diff.shape)

        temp_pred[temp_diff < a] = -1

        temp_pred[temp_diff > a] = 1

        temp_pred[(temp_diff == a)] = 0



        res = accuracy_score(true_result, temp_pred)

        results_a.append(res)

        

    d[league].append(np.max(results_a))
for league in matches.league_id.unique():

    matches_of_league = matches[matches['league_id'] == league]

    

    temp_diff = np.array(np.int64((matches_of_league.points_home - matches_of_league.points_away)))

    temp_pred = np.array(np.int64(np.sign(matches_of_league.points_home - matches_of_league.points_away)))

    result_if_home_team_always_win = np.ones(matches_of_league.shape[0])

    true_result = np.array(matches_of_league.result)

    

    list_a = sorted(list(set(temp_diff)))

    results = list()



    for i, a in enumerate(list_a):

        temp_pred = np.zeros(temp_diff.shape)

        temp_pred[temp_diff < a] = -1

        temp_pred[temp_diff >= a] = 1

        res = accuracy_score(true_result, temp_pred)

        results.append(res)        



    d[league].append(np.max(results))
accuracy = list(d.items())

keys = list(map(lambda x: x[0], accuracy))

values = list(map(lambda x: x[1], accuracy))

df = pd.DataFrame(values)

df.index = list(leagues.loc[keys]['name'])

df.columns = ['hometeam_always_win', 'points_difference', 'points_difference_without_drawns']

df