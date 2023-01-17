!pip install -Iv pulp==1.6.8 --quiet
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

import os

import time

import scipy.stats as stats

import pulp

import unidecode
## Read data

# Players df contains the summary of player performance from each season
# Each row represents one player

## GWs df contains the information of each player for each gameweek
## Each row represents a player's performance for a gameweek

folderpath = '../input/fantasypremierleague/'

players_1617_df = pd.read_csv(folderpath+'players_raw_1617.csv')
players_1718_df = pd.read_csv(folderpath+'players_raw_1718.csv')
players_1819_df = pd.read_csv(folderpath+'players_raw_1819.csv')
players_1920_df = pd.read_csv(folderpath+'players_raw_1920.csv')

gws_1617_df = pd.read_csv(folderpath+'merged_gws_1617.csv')
gws_1718_df = pd.read_csv(folderpath+'merged_gws_1718.csv',encoding='latin')
gws_1819_df = pd.read_csv(folderpath+'merged_gw_1819.csv',encoding='latin')
gws_1920_df = pd.read_csv(folderpath+'merged_gw_1920.csv',engine='python')

team_codes_df = pd.read_csv(folderpath+'teams.csv')

# Clean the headers to be used later
team_codes_df.columns.values[2:] = team_codes_df.columns[2:].str.replace('team_', '')
# remove Danny Wards from 18/19 season
players_1819_df = players_1819_df[((players_1819_df.first_name == "Danny") & (players_1819_df.second_name=="Ward"))==False]
gws_1819_df = gws_1819_df[gws_1819_df.name.str.contains("Danny_Ward")==False]
player_df_list = [players_1617_df, players_1718_df, players_1819_df, players_1920_df]
gw_df_list = [gws_1617_df, gws_1718_df, gws_1819_df, gws_1920_df]
# append season and season index to dfs

seasons = ['1617', '1718', '1819', '1920']
season_nums = list(range(len(seasons)))

for i in range(len(seasons)):
    
    player_df_list[i]['season'] = seasons[i]
    gw_df_list[i]['season'] = seasons[i]
    
    player_df_list[i]['season_num'] = season_nums[i]
    gw_df_list[i]['season_num'] = season_nums[i]
# combine dataframes from all seasons into one

players_df = pd.concat(player_df_list)
gws_df = pd.concat(gw_df_list)

players_df.reset_index(inplace=True)
gws_df.reset_index(inplace=True)
## Get full name
# Cleans up accents and also makes processing easier

def get_full_name_playerdf(first_name, second_name):
    full_name = first_name +'_' + second_name
    full_name = full_name.replace(" ", "_")
    full_name = full_name.replace("-", "_")
    full_name = unidecode.unidecode(full_name)
    
    return full_name


# Translate player positions into string for easier readability

positions_dict = {
    1: 'Keeper',
    2: 'Defender',
    3: 'Midfielder',
    4: 'Forward'
    
}


players_df['full_name'] = players_df.apply(lambda x: get_full_name_playerdf(x.first_name, x.second_name), axis=1).str.lower()
players_df['position'] = players_df.element_type.map(positions_dict)
players_df['starting_cost'] = players_df.now_cost - players_df.cost_change_start_fall
players_df['cost_bin'] = players_df.now_cost.apply(lambda x: np.floor(x/10))



gws_df['full_name'] = gws_df.name.str.replace('_\d+','')
gws_df['full_name'] = gws_df['full_name'].str.replace(" ", "_").str.replace("-", "_").str.replace('_\d+','')
gws_df['full_name'] = gws_df['full_name'].apply(lambda x: unidecode.unidecode(x))
gws_df['full_name'] = gws_df['full_name'].str.lower()

    
def clean_gw_df(player_df, gw_df, team_codes_df):
    
    # Returns a df with player position, player's team name, and opponent's team name
    
    pdf = player_df.copy()[['full_name', 'season', 'position', 'player_team_name']]
    gdf = gw_df.copy()
    
    gdf = gdf.merge(pdf, on=['full_name', 'season'], how='left')
    
    
    dfs = []
    for s, group in gdf.groupby('season'):

        temp_code_df = team_codes_df[['team', s]]
        temp_code_df = temp_code_df.dropna()
        
        group = group[['opponent_team']]
        group['opponent_team_name'] = group.opponent_team.map(temp_code_df.set_index(s).team)
        dfs.append(group[['opponent_team_name']])
        
    out_df = pd.concat(dfs, axis=0)
    out_df = pd.concat([gdf, out_df], axis=1)

    return out_df


gws_df.opponent_team = gws_df.opponent_team.astype(float)
players_df['player_team_name'] = players_df.team_code.map(team_codes_df.set_index('team_code').team)
gws_df = clean_gw_df(players_df, gws_df, team_codes_df)
def make_available_players_df(this_season_player_df, last_season_player_df):
    
    
    last_season_player_df = last_season_player_df[last_season_player_df.minutes > 0]
    last_season_player_df = last_season_player_df[['full_name', "total_points"]]
    last_season_player_df.rename(columns={'total_points': "total_points_last_season"},
                                inplace=True)
    
    available_players_df = pd.merge(this_season_player_df,
                                    last_season_player_df,
                                   on='full_name', how='left')
    
    available_players_df.total_points_last_season = available_players_df.groupby(['position', 'cost_bin']).total_points_last_season.transform(lambda x: x.fillna(x.mean()))
    
    return available_players_df
current_season_player_df = players_df[players_df.season=='1920'] 
previous_season_player_df = players_df[players_df.season=='1819'] 

available_players_df = make_available_players_df(current_season_player_df, previous_season_player_df)
def get_cheapest_players(player_df):
    
    cheapest_player_names = []
    total_cost = 0
    
    # for each position, sort the players by cost (in ascending order)
    # then, get the player with the most number of points
    
    for position, group in player_df.groupby('position'):
        cheapest_players =  group[(group.starting_cost == group.starting_cost.min())]
        top_cheapest_player = cheapest_players[cheapest_players.total_points == cheapest_players.total_points.max()]
        
        cheapest_player_name = top_cheapest_player.full_name.values[0]
        
        cheapest_player_names += [cheapest_player_name]
        total_cost += top_cheapest_player.starting_cost.values[0]
        
        print(position, ": ", cheapest_player_name )
        
    return cheapest_player_names, total_cost
bench_players, bench_cost = get_cheapest_players(available_players_df)
def make_decision_variables(player_df):
    return [pulp.LpVariable(i, cat="Binary") for i in player_df.full_name]
def make_optimization_function(player_df, decision_variables):
    op_func = ""

    for i, player in enumerate(decision_variables):
        op_func += player_df.total_points_last_season[i]*player
        
    return op_func
def make_cash_constraint(player_df, decision_variables, available_cash):
    total_paid = ""
    for rownum, row in player_df.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                formula = row['starting_cost']*player
                total_paid += formula

    return (total_paid <= available_cash)
def make_player_constraint(position, n, decision_variables, player_df):
    
    total_n = ""
    
    player_positions = player_df.position
    
    for i, player in enumerate(decision_variables):
        if player_positions[i] == position:
            total_n += 1*player
            
    return(total_n == n)
def add_team_constraint(prob, player_df, decision_variables):

    for team, group in player_df.groupby('team_code'):
        team_total = ''
        
        for player in decision_variables:
            if player.name in group.full_name.values:
                formula = 1*player
                team_total += formula
                
        
        prob += (team_total <= 3)
available_cash = 1000 - bench_cost

prob = pulp.LpProblem('InitialTeam', pulp.LpMaximize)

decision_variables = make_decision_variables(available_players_df)
prob += make_optimization_function(available_players_df, decision_variables)
prob += make_cash_constraint(available_players_df, decision_variables, available_cash)
prob += make_player_constraint("Keeper", 1, decision_variables, available_players_df) 
prob += make_player_constraint("Defender", 4, decision_variables, available_players_df) 
prob += make_player_constraint("Midfielder", 4, decision_variables, available_players_df) 
prob += make_player_constraint("Forward", 2, decision_variables, available_players_df)

add_team_constraint(prob, available_players_df, decision_variables)
## Solve

prob.writeLP('InitialTeam.lp')
optimization_result = prob.solve()

## Get initial team

def get_initial_team(prob, player_df):
    
    variable_names = [v.name for v in prob.variables()]
    variable_values = [v.varValue for v in prob.variables()]

    initial_team = pd.merge(pd.DataFrame({'full_name': variable_names,
                  'selected': variable_values}),
                                       player_df, on="full_name")
    
    initial_team = initial_team[initial_team.selected==1.0] 
    
    return initial_team

    
initial_team_df = get_initial_team(prob, available_players_df)
initial_team_df[['full_name', "position", "starting_cost", "player_team_name"]]
## Sanity check

def sanity_check(team_df):
    print('Sanity check for starting 11: ')
    print('*'*88) 
    
    print('Number of players in each position: ')
    for pos, group in team_df.groupby('position'):
        print(pos, ': ', len(group), sep='')
        
    
    print('*'*88)   
    print('Number of players from each team: ')
    print(team_df.groupby('player_team_name').position.count())
    
    print('*'*88)    
    print('Total cost:', team_df.starting_cost.sum())
    

sanity_check(initial_team_df)
captain = get_initial_team(prob, previous_season_player_df).sort_values("total_points", ascending=False).head(1).full_name.values[0]
captain = get_initial_team(prob, previous_season_player_df).sort_values("total_points", ascending=False).head(1).full_name.values[0]

total_points = current_season_player_df[current_season_player_df.full_name.isin(initial_team_df.full_name)].total_points.sum()
total_points += current_season_player_df[current_season_player_df.full_name==captain].total_points

print("Total points for 19/20 season:", total_points.values[0])
def get_team_points(was_home, h_score, a_score):
    
    if h_score == a_score:
        return 1
    
    if h_score > a_score:
        if was_home:
            return 3
        else: 
            return 0
    
    if h_score < a_score:
        if was_home:
            return 0
        else: 
            return 3
def get_opponent_points(team_points):
    if team_points == 1:
        return 1
    
    if team_points == 3:
        return 0
    
    if team_points == 0:
        return 3
gws_df['team_points']= gws_df.apply(lambda x: get_team_points(x.was_home, x.team_h_score, x.team_a_score), axis=1)
gws_df['opponent_points'] = gws_df.team_points.apply(lambda x: get_opponent_points(x))
def player_lag_features(gw_df, features, lags):
    
    out_df = gw_df.copy()
    lagged_features = []
    
    for feature in features:
            
        for lag in lags:
            
            lagged_feature = 'last_' + str(lag) + '_' + feature
            
            if lag == 'all':
                out_df[lagged_feature] = out_df.sort_values('round').groupby(['season', 'full_name'])[feature]\
            .apply(lambda x: x.cumsum() - x)
                
            else:

                out_df[lagged_feature] = out_df.sort_values('round').groupby(['season', 'full_name'])[feature]\
                .apply(lambda x: x.rolling(min_periods=1, window=lag+1).sum() - x)

            lagged_features.append(lagged_feature)
    
    return out_df, lagged_features
def team_lag_features(gw_df, features, lags):
    out_df = gw_df.copy()
    lagged_features = []
    
    for feature in features:

        ## Create a df for each feature
        ## Then, self-join so that the opponent info for that feature is included
        ## Then, create lagged features and join the columns to the feature df
        ## Do the same for the opponent feature
        ## Exit loop, merge with the original df
        
        feature_name = feature + '_team'
        opponent_feature_name = feature_name + '_opponent'
        
  
        feature_team = out_df.groupby(['player_team_name', 'season', 'round', 'kickoff_time', 'opponent_team_name'])\
                        [feature].max().rename(feature_name).reset_index()
        
        # self join to get opponent info
        
        feature_team = feature_team.merge(feature_team,
                          left_on=['player_team_name', 'season', 'round', 'kickoff_time', 'opponent_team_name'],
                          right_on=['opponent_team_name', 'season', 'round', 'kickoff_time', 'player_team_name'],
                          how='left',
                          suffixes=('', '_opponent'))
            
        

        
        for lag in lags:
            lagged_feature_name = 'last_' + str(lag) + '_' + feature_name
            lagged_opponent_feature_name = 'opponent_last_' + str(lag) + '_' + feature
            

            if lag == 'all':
                
                feature_team[lagged_feature_name] = feature_team.sort_values('round').groupby('player_team_name')[feature_name]\
                                                .apply(lambda x: x.cumsum() - x)
            
                feature_team[lagged_opponent_feature_name] = feature_team.groupby('player_team_name')[opponent_feature_name]\
                                                .apply(lambda x: x.cumsum() - x)
            else:
                    
                       
                feature_team[lagged_feature_name] = feature_team.sort_values('round').groupby('player_team_name')[feature_name]\
                                                    .apply(lambda x: x.rolling(min_periods=1,
                                                                              window=lag+1).sum()-x)

                feature_team[lagged_opponent_feature_name] = feature_team.groupby('player_team_name')[opponent_feature_name]\
                                                    .apply(lambda x: x.rolling(min_periods=1,
                                                                              window=lag+1).sum()-x)

            lagged_features.extend([lagged_feature_name, lagged_opponent_feature_name])
            
        out_df = out_df.merge(feature_team,
                             on=['player_team_name', 'season', 'round', 'kickoff_time', 'opponent_team_name'],
                             how='left')
        
        
        return out_df, lagged_features
player_features_to_lag = [
    'assists',
     'bonus',
     'bps',
     'creativity',
     'clean_sheets',
     'goals_conceded',
     'goals_scored',
     'ict_index',
     'influence',
     'minutes',
     'threat']

team_features_to_lag = ['goals_conceded', 'goals_scored', 'team_points', 'opponent_points']
lagged_gw_df_players, lagged_player_features = player_lag_features(gws_df, player_features_to_lag, ['all', 1, 3, 5])
lagged_gw_df, lagged_team_features = team_lag_features(lagged_gw_df_players, team_features_to_lag, ['all', 1, 3, 5])
relevant_features = ['position', 'was_home', 'minutes', 'value', 'round', 'season_num'] + \
    lagged_player_features + \
    lagged_team_features 
def make_dummies(df, numerical_features, categorical_features):
    
  
    X_num = df[numerical_features]
    X_cat = df[categorical_features]
    
    X_cat = X_cat.astype(str)
    X_cat = pd.get_dummies(X_cat)
    
    # Join categorical and numerical features
    X = pd.concat([X_num, X_cat], axis=1)
    
    return X
categorical_features = ['was_home', 'position']
numerical_features = numerical_features = list(set(relevant_features) - set(categorical_features))
train_df = lagged_gw_df[(lagged_gw_df.season!='1920')]
test_df = lagged_gw_df[(lagged_gw_df.season=='1920') ]


# XGBoost handles NA values, but the other scikit learn methods (that I've chosen) do not

lagged_gw_df_no_na = lagged_gw_df.dropna(subset=relevant_features + ['total_points', 'season'])
train_df_no_na = lagged_gw_df_no_na[lagged_gw_df_no_na.season!='1920']
test_df_no_na = lagged_gw_df_no_na[lagged_gw_df_no_na.season=='1920']
X_train = make_dummies(train_df[relevant_features], numerical_features, categorical_features)
y_train = train_df.total_points

X_train_no_na = make_dummies(train_df_no_na[relevant_features], numerical_features, categorical_features)
y_train_no_na = train_df_no_na.total_points
X_test = make_dummies(test_df, numerical_features, categorical_features)
y_test = test_df.total_points

X_test_no_na = make_dummies(test_df_no_na, numerical_features, categorical_features)
y_test_no_na = test_df_no_na.total_points
params = {
         'max_depth': list(range(3,7)),  
    'min_child_weight': list(range(10,51)),
    'learning_rate':  [0.03, 0.15, 0.3, 0.45, 0.6],
    'subsample': stats.uniform(0.8, 0.1),
    'colsample_bytree': [0.8, 0.1]}

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
xgb_reg.fit(X_train, y_train)



xgb_cv = RandomizedSearchCV(xgb_reg, params, cv=3, scoring='neg_root_mean_squared_error',
                            random_state=999)

xgb_cv.fit(X_train, y_train)
xgb_best = xgb.XGBRegressor(objective='reg:squarederror')
xgb_best.set_params(**xgb_cv.best_params_)

# Initialize the models. I am doing it this way as predictions are easy to calculate so I will not be storing them until I need them


seed = 999
models = []
models.append(('LinReg', LinearRegression()))
models.append(('LassoReg', LassoCV()))
models.append(('RidgeReg', RidgeCV()))

def get_cv_scores(models, X, y, k=5, seed=999):
    
    # inspired by the excellent tutorial by Jason Brownlee:
    # https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
    
    names = []
    results = []
    print("Cross val scores:")
    
    for name, m in models:
        cv_results = -cross_val_score(m, X, y, cv=k, scoring='neg_root_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std() ))
        print("")
        print("*"*88)
        print("")
        
    return names, results

model_names, model_results = get_cv_scores(models, X_train_no_na, y_train_no_na)
xgb_cv_scores = get_cv_scores([("XGB", xgb_best)], X_train, y_train)
model_names += xgb_cv_scores[0]
model_results += xgb_cv_scores[1]
def compare_model_scores(model_names, model_results):
    
    fig = plt.figure()
    fig.suptitle('Model comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(model_results)
    ax.set_xticklabels(model_names)
    plt.show()
compare_model_scores(model_names, model_results)
def make_predicted_table(y_test, y_pred, gw_df):
    results_df = pd.DataFrame(list(zip(y_test.tolist(), y_pred.tolist())),
                             columns=["actual", "predicted"])
    
    
    results_df.reset_index(drop=True, inplace=True)
    gw_df.reset_index(inplace=True)
    pred_df = pd.concat([gw_df, results_df], axis=1)
    
    return pred_df
def get_suggested_transfer(predicted_df, team_list, current_money):
    
    predicted_diff = 0
    money_change = 0
    suggested_in = ''
    suggested_out = ''
    team_df = predicted_df[(predicted_df.full_name.isin(team_list))]
    
  
    teams_dict = {}
    for i, row in team_df.iterrows():
        if row.player_team_name not in teams_dict:
            teams_dict[row.player_team_name] = [row.full_name]
        else:
            teams_dict[row.player_team_name].append(row.full_name)
            
            
    for position in ["Defender", "Midfielder", "Forward"]:
        
        # don't bother about keepers, variance in scores is not that great
        # so, save the free transfer for other positions
       
        
        
        player_df = predicted_df[predicted_df.position==position].sort_values('predicted', ascending=False).reset_index()

        
        lowest_pos = 0
        player_names = team_df[team_df.position==position].full_name.values
        
        # loop through the players for this position, and get the rank (row number) of the player with the lowest predicted score
        for p in player_names:
            player_pos = player_df[player_df.full_name==p].index[0]
            if player_pos > lowest_pos:
                lowest_pos = player_pos
                potential_out = p
                potential_out_cost = team_df[team_df.full_name==p].value.values[0]
                potential_out_team = team_df[team_df.full_name==p].player_team_name.values[0]
                
            elif len(player_names) <= 1:
                potential_out_cost = 0
                potential_out_team = 'none'
                potential_out = 'none'
                
        # get all players above this player
        potential_players = player_df[:lowest_pos]
        
        
        # only keep players that we can afford
        potential_players = potential_players[potential_players.value <= potential_out_cost + current_money]
        
        # only keep players who played (need a better way of doing this)
        potential_players = potential_players[potential_players.minutes > 0]

        # get the prediction difference for each suggested player
        # select the one with the highest difference as the suggested transfer (compare across positons)
        
        potential_out_predicted = team_df[team_df.full_name==p].predicted.values[0]
        
        for i, row in potential_players.iterrows():
                
            # skip if it is a player we already have
            if row.full_name in team_list:
                continue



            # if there are no other players of the same team, it's ok to consider this player
            # if not, check whether there are 3 players of the same team already
            if row.player_team_name not in teams_dict:
                pass
            else:
                if len(teams_dict[row.player_team_name]) == 3:
                    # if there are already 3 players of the same team,
                    # can't take another player of the same team
                    # unless the suggested_out is the same team as suggested_in (direct swap)
                   
                    if row.player_team_name == potential_out_team:
                        pass
                    else:
                        continue
                else:
                    pass
                
            
            # check for difference in predictions
            if row.predicted - potential_out_predicted > predicted_diff:
                predicted_diff = row.predicted - potential_out_predicted
                suggested_in = row.full_name
                suggested_out = potential_out
                
                # calculate change in money
                money_change = potential_out_cost - row.value
                
    return suggested_in, suggested_out, money_change
def get_score(team_list, gw_df):
    
    gw_score = gw_df[gw_df.full_name.isin(team_list)].actual.sum() \
        + gw_df[(gw_df.full_name.isin(team_list)) & (gw_df.position!='Keeper')].sort_values("predicted", ascending=False).head(1).actual.values[0]
        
    
    return gw_score
def get_performance(team_list, starting_money, gw_list,
                   prediction_df):
    
    current_money = starting_money
    total_score = 0
    
    
    in_list = []
    out_list = []
    score_list = []
    unplayed_list = []
    
    
    for gw in gw_list:

        gw_df = prediction_df[prediction_df.GW==gw]
        money_change = 0
        suggested_in = ''
        suggested_out = ''
        if gw > 1:
            

            suggested_in, suggested_out, money_change = get_suggested_transfer(gw_df, team_list, current_money)
        
            current_money += money_change

            team_list.append(suggested_in)
            team_list.remove(suggested_out)
            
        

        ## Calculate scores
        
        gw_score = get_score(team_list, gw_df)

        
        out_list.append(suggested_out)
        in_list.append(suggested_in)
        score_list.append(gw_score)
        
        total_score += gw_score
        
    out_df = pd.DataFrame({'GW': gw_list,
                          'player_in': in_list,
                          'player_out': out_list,
                          'total_score': score_list})
    
    print(total_score)
    
    return(out_df)
## Get predictions for the two best models, Linear Regression and XGBoost

lin_reg = LinearRegression()
lin_reg.fit(X_train_no_na, y_train_no_na)
linreg_predictions = lin_reg.predict(X_test_no_na)


xgb_best.fit(X_train, y_train)
xgb_predictions = xgb_best.predict(X_test)
predicted_df_lin_reg = make_predicted_table(y_test_no_na, linreg_predictions, test_df_no_na[relevant_features + ['full_name', 'GW', 'player_team_name', 'total_points']] )
predicted_df_xgb = make_predicted_table(y_test, xgb_predictions, test_df[relevant_features + ['full_name', 'GW', 'player_team_name', 'total_points']] )
captain = get_initial_team(prob, previous_season_player_df).sort_values("total_points", ascending=False).head(1).full_name.values[0]

total_points = current_season_player_df[current_season_player_df.full_name.isin(initial_team_df.full_name)].total_points.sum()
total_points += current_season_player_df[current_season_player_df.full_name==captain].total_points

print("Total points for 19/20 season:", total_points.values[0])
## Linear regression

my_team = list(initial_team_df.full_name)
gameweeks = (test_df.GW).unique()
starting_money = 1000 - bench_cost - initial_team_df.starting_cost.sum()

lin_reg_perf = get_performance(my_team, starting_money, gameweeks,
                   predicted_df_lin_reg)
## XGBoost


my_team = list(initial_team_df.full_name)
gameweeks = (test_df.GW).unique()
starting_money = 1000 - bench_cost - initial_team_df.starting_cost.sum()

xgb_cv_perf = get_performance(my_team, starting_money, gameweeks,
                   predicted_df_xgb)
xgb_cv_perf
stats.ttest_rel(xgb_cv_perf.total_score, lin_reg_perf.total_score)
