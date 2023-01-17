# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

print(os.listdir("../input/fifa-international-soccer-mens-ranking-1993now"))
print(os.listdir("../input/international-football-results-from-1872-to-2017"))

# Any results you write to the current directory are saved as output.
def result(row):
    if row['home_score'] > row['away_score']: return 1 
    elif row['home_score'] < row['away_score']: return 2
    else: return 0

def tournament_importance(row):
    if row == 'FIFA World Cup': return 100
    elif row == 'UEFA Euro' or row == 'African Cup of Nations' or row == 'African Nations Championship' or row == 'Copa América' or row == 'Gold Cup' or row == 'AFC Asian Cup': return 75
    elif row == 'Confederations Cup': return 60
    elif 'qualification' in row: return 50
    elif row == 'Friendly': return 1
    else: return 10
    
def winner(row):
    if row['home_score'] > row['away_score']: return row['home_team'] 
    elif row['home_score'] < row['away_score']: return row['away_team']
    else: return 'DRAW'
    
def loser(row):
    if row['home_score'] < row['away_score']: return row['home_team'] 
    elif row['home_score'] > row['away_score']: return row['away_team']
    else: return 'DRAW'
    
#Generic function for making a classification model and accessing performance:
def class_model(model, data, predictors, outcome):
    model.fit(data[predictors],data[outcome])
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print('Cross validation Score : %s' % '{0:.3%}'.format(np.mean(error)))
    model.fit(data[predictors],data[outcome])
    
def get_match_result(model_home, model_away, home, away, predictor_var):
    row_home = pd.DataFrame(
        np.array([[
            world_cup_rankings.loc[home, 'rank'], 
            country_info.loc[home, 'nb_goals'], 
            country_info.loc[home, 'goal_avg'], 
            country_info.loc[home, 'nb_matches'], 
            country_info.loc[home, 'wins'], 
            country_info.loc[home, 'loses'],
            country_info.loc[home, 'draws'],
            world_cup_rankings.loc[away, 'rank'],
            country_info.loc[away, 'nb_goals'], 
            country_info.loc[away, 'goal_avg'], 
            country_info.loc[away, 'nb_matches'], 
            country_info.loc[away, 'wins'], 
            country_info.loc[away, 'loses'],
            country_info.loc[away, 'draws'],
            100]]), columns=predictor_var)
    
    row_away = pd.DataFrame(
        np.array([[
            world_cup_rankings.loc[away, 'rank'], country_info.loc[away, 'nb_goals'], country_info.loc[away, 'goal_avg'], country_info.loc[away, 'nb_matches'], country_info.loc[away, 'wins'], country_info.loc[away, 'loses'], country_info.loc[away, 'draws'],
            world_cup_rankings.loc[home, 'rank'], country_info.loc[home, 'nb_goals'], country_info.loc[home, 'goal_avg'], country_info.loc[home, 'nb_matches'], country_info.loc[home, 'wins'], country_info.loc[home, 'loses'], country_info.loc[home, 'draws'],
            100]]), columns=predictor_var)
        
    return (
            pd.concat((pd.DataFrame(model_home.predict_proba(row_home)), pd.DataFrame(model_away.predict_proba(row_away)))).mean().idxmax(),
            pd.concat((pd.DataFrame(model_away.predict_proba(row_home)), pd.DataFrame(model_home.predict_proba(row_away)))).mean().idxmax(),
            pd.concat((pd.DataFrame(model_home.predict_proba(row_home)), pd.DataFrame(model_away.predict_proba(row_away)))).mean().max(),
            pd.concat((pd.DataFrame(model_away.predict_proba(row_home)), pd.DataFrame(model_home.predict_proba(row_away)))).mean().max()
        )
matches = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
rankings = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')

matches['year'], matches['month'], matches['day'] = matches['date'].str.split('-', 2).str
matches['date'] = pd.to_datetime(matches['date'])
matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches = matches.loc[matches['date'] >= "1993-08-08"]

matches['result'] = matches.apply(lambda row: result(row), axis=1)
matches['scores'] = zip(matches['home_score'], matches['away_score'])
    
matches['tournament_importance'] = matches.apply(lambda row: tournament_importance(row['tournament']), axis=1)
matches['tournament_importance'].value_counts()
    
matches['winner'] = matches.apply(lambda row: winner(row), axis=1)
matches['loser'] = matches.apply(lambda row: loser(row), axis=1)

winners = matches.groupby('winner').size().reset_index(name='wins').sort_values(by='wins', ascending=False)
winners = winners.rename(columns={'winner':'team'})
winners = winners[winners.team != 'DRAW']

losers = matches.groupby('loser').size().reset_index(name='loses').sort_values(by='loses', ascending=False)
losers = losers.rename(columns={'loser':'team'})
losers = losers[losers.team != 'DRAW']

# create two dataframe for the home and away teams
home = matches[['home_team', 'home_score']].rename(columns={'home_team':'team', 'home_score':'score'})
away = matches[['away_team', 'away_score']].rename(columns={'away_team':'team', 'away_score':'score'})
# merge it into one
team_score = home.append(away).reset_index(drop=True)

# make an aggregation of the the score column group by the team
country_info = team_score.groupby('team')['score'].agg(['sum','count','mean']).reset_index()
country_info = country_info.rename(columns={'sum':'nb_goals', 'count':'nb_matches', 'mean':'goal_avg'})
country_info = country_info.sort_values(by='nb_matches', ascending=False)

country_info = country_info.merge(winners, on='team')
country_info = country_info.merge(losers, on='team')
country_info['draws'] = country_info['nb_matches'] - country_info['wins'] - country_info['loses']
country_info = country_info.set_index(['team'])

matches = matches.merge(country_info, 
                        left_on=['home_team'], 
                        right_on=['team'])
matches = matches.merge(country_info, 
                        left_on=['away_team'], 
                        right_on=['team'], 
                        suffixes=('_home', '_away'))


rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'total_points', 'cur_year_avg', 'cur_year_avg_weighted', 'rank_date']]
rankings = rankings.replace({"IR Iran": "Iran"})
rankings['year'], rankings['month'], rankings['day'] = rankings['rank_date'].str.split('-', 2).str
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

rankings = rankings.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()

# join the ranks
matches = matches.merge(rankings, 
                        left_on=['date', 'home_team'], 
                        right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings, 
                        left_on=['date', 'away_team'], 
                        right_on=['rank_date', 'country_full'], 
                        suffixes=('_home', '_away'))

matches.shape
country_info.head()
matches.head()
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])
# world_cup_rankings.loc[(world_cup_rankings.country_full.str.startswith('I'))]
predictor_var = ['rank_home', 
    'nb_goals_home', 'nb_matches_home', 'goal_avg_home', 'wins_home', 'loses_home', 'draws_home',
    'rank_away', 
    'nb_goals_away', 'nb_matches_away', 'goal_avg_away', 'wins_away', 'loses_away', 'draws_away',
    'tournament_importance']

rdcHome = RandomForestClassifier(n_estimators=100)
rdcAway = RandomForestClassifier(n_estimators=100)

class_model(rdcHome, matches, predictor_var, ['home_score'])
class_model(rdcAway, matches, predictor_var, ['away_score'])

# LR_clf = LogisticRegression()
# LR_clf.fit(X_train, y_train)
# y_pred = LR_clf.predict(X_test)
# accuracy_score(y_test, y_pred)
print('Portugal', 'Spain', get_match_result(rdcHome, rdcAway, 'Portugal', 'Spain', predictor_var))
print("--------------")

print('Denmark', 'Australia', get_match_result(rdcHome, rdcAway, 'Denmark', 'Australia', predictor_var))
print("--------------")

print('France', 'Peru', get_match_result(rdcHome, rdcAway, 'France', 'Peru', predictor_var))
print("--------------")

print('Argentina', 'Croatia', get_match_result(rdcHome, rdcAway, 'Argentina', 'Croatia', predictor_var))
print("--------------")

print('Iceland', 'Nigeria', get_match_result(rdcHome, rdcAway, 'Iceland', 'Nigeria', predictor_var))
print("--------------")

print('Brazil', 'Costa Rica', get_match_result(rdcHome, rdcAway, 'Brazil', 'Costa Rica', predictor_var))
print("--------------")

print('Brazil', 'Switzerland', get_match_result(rdcHome, rdcAway, 'Brazil', 'Switzerland', predictor_var))
print("--------------")

print('Switzerland', 'Serbia', get_match_result(rdcHome, rdcAway, 'Switzerland', 'Serbia', predictor_var))
groupA = ['Russia', 'Saudi Arabia', 'Egypt', 'Uruguay']
groupB = ['Portugal', 'Spain', 'Morocco', 'Iran']
groupC = ['France', 'Australia', 'Peru', 'Denmark']
groupD = ['Argentina', 'Iceland', 'Croatia', 'Nigeria']
groupE = ['Brazil', 'Switzerland', 'Costa Rica', 'Serbia']
groupF = ['Germany', 'Mexico', 'Sweden', 'Korea Republic']
groupG = ['Belgium', 'Panama', 'Tunisia', 'England']
groupH = ['Poland', 'Senegal', 'Colombia', 'Japan']
groups = [groupA, groupB, groupC, groupD, groupE, groupF, groupG, groupH]
from itertools import combinations
from IPython.display import display, HTML

groups_result = []

for group in groups:
    ranking = pd.DataFrame({'points':[0,0,0,0], 'diff':[0,0,0,0], 'goals':[0,0,0,0]}, index=group)
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(group, 2):
        score = get_match_result(rdcHome, rdcAway, home, away, predictor_var)
        print("{} vs. {}: ".format(home, away), score)
        
        if score[0] > score[1]:
            ranking.loc[home, 'points'] += 3
            ranking.loc[home, 'goals'] += score[0]
            ranking.loc[away, 'goals'] += score[1]
            ranking.loc[home, 'diff'] += score[0] - score[1]
            ranking.loc[away, 'diff'] -= score[0] - score[1]
        elif score[0] < score[1]:
            ranking.loc[away, 'points'] += 3
            ranking.loc[away, 'goals'] += score[1]
            ranking.loc[home, 'goals'] += score[0]
            ranking.loc[away, 'diff'] += score[1]-score[0]
            ranking.loc[home, 'diff'] -= score[1]-score[0]
        else:
            ranking.loc[[home, away], 'points'] += 1
            ranking.loc[[home, away], 'goals'] += score[0]
            
    groups_result.append(ranking.sort_values(by=['points','diff','goals'], ascending=False))
for group_rank in groups_result:
    display(group_rank)
def team_winner(team1, team2):
    score = get_match_result(rdcHome, rdcAway, team1, team2, predictor_var)
    
    if score[0] > score[1]:
        result = (team1, team2)
    elif score[1] > score[0]:
        result = (team2, team1)
    elif score[2] > score[3]:
        result = (team1, team2)
    elif score[3] > score[2]:
        result = (team2, team1)
    else:
        print('ERROR')
    
    print("{} vs. {}: ".format(team1, team2), (score[0], score[1]), result[0])
    return result

round_of_16 = []
quarter_finals = []
semi_finals = []

print("___1/8 final___")
for i in range(0, 8, 2):
    round_of_16.append(team_winner(groups_result[i].index[0], groups_result[i+1].index[1]))
    round_of_16.append(team_winner(groups_result[i].index[1], groups_result[i+1].index[0]))

print("___1/4 final___")
quarter_finals.append(team_winner(round_of_16[0][0], round_of_16[2][0]))
quarter_finals.append(team_winner(round_of_16[1][0], round_of_16[3][0]))
quarter_finals.append(team_winner(round_of_16[4][0], round_of_16[6][0]))
quarter_finals.append(team_winner(round_of_16[5][0], round_of_16[7][0]))

print("___1/2 final___")
semi_finals.append(team_winner(quarter_finals[0][0], quarter_finals[2][0]))
semi_finals.append(team_winner(quarter_finals[1][0], quarter_finals[3][0]))

print("___Little final___")
little_final = team_winner(semi_finals[0][1], semi_finals[1][1])

print("___Final___")
final = team_winner(semi_finals[0][0], semi_finals[1][0])
winners = matches.groupby('winner').size().reset_index(name='wins').sort_values(by='wins', ascending=False)
winners = winners.rename(columns={'winner':'team'})
winners = winners[winners.team != 'DRAW']
winners.head(10)
losers = matches.groupby('loser').size().reset_index(name='loses').sort_values(by='loses', ascending=False)
losers = losers.rename(columns={'loser':'team'})
losers = losers[losers.team != 'DRAW']
losers.head(10)
team_wins = country_info[['team', 'wins']].sort_values(by='wins', ascending=False)
team_wins.head(10)
team_loses = country_info[['team', 'loses']].sort_values(by='loses', ascending=False)
team_loses.head(10)
team_draws = country_info[['team', 'draws']].sort_values(by='draws', ascending=False)
team_draws.head(10)
tournament = matches.groupby('tournament').size().reset_index(name='nb_matches').sort_values(by='nb_matches', ascending=False)
tournament.head(10)