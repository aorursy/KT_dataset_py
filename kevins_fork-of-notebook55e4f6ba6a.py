# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
data = pd.read_csv('../input/data.csv')

data.set_index('shot_id', inplace=True)
unknown_mask = data['shot_made_flag'].isnull()

data_cl = data.copy()

target = data_cl['shot_made_flag'].copy()
data_cl.drop('team_id', inplace=True, axis=1) #only 1 category

data_cl.drop('lat', inplace=True, axis=1) # correlated with loc_x

data_cl.drop('lon', inplace=True, axis=1) # correlated with loc_y

data_cl.drop('game_id', inplace=True, axis=1) # should not be dependent on game id, furthermore it's contained in opponent/match

data_cl.drop('game_event_id', inplace=True, axis=1) # independent, unique for every shots in a game

data_cl.drop('team_name', inplace=True, axis=1) # always LA Lakers

data_cl.drop('shot_made_flag', inplace=True, axis=1) # target variables
# time into the game

data_cl['seconds_from_period_end'] = 60 * data_cl['minutes_remaining'] + data_cl['seconds_remaining']

data_cl['last_5_sec_in_period'] = data_cl['seconds_from_period_end'] < 5

data_cl['seconds_from_period_start'] = 60*(11-data_cl['minutes_remaining'])+(60-data_cl['seconds_remaining'])

data_cl['seconds_from_game_start'] = (data_cl['period'] <= 4).astype(int)*(data_cl['period']-1)*12*60 + (data_cl['period'] > 4).astype(int)*((data_cl['period']-4)*5*60 + 3*12*60) + data_cl['seconds_from_period_start']



# drop redundant features

data_cl.drop('minutes_remaining', axis=1, inplace=True)

data_cl.drop('seconds_remaining', axis=1, inplace=True)
data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')

data_cl.drop('matchup', axis=1, inplace=True)
data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])



# year and month

data_cl['game_year'] = data_cl['game_date'].dt.year

data_cl['game_month'] = data_cl['game_date'].dt.month



# day of week/year

data_cl['dayOfWeek'] = data_cl['game_date'].dt.dayofweek

data_cl['dayOfYear'] = data_cl['game_date'].dt.dayofyear



data_cl.drop('game_date', axis=1, inplace=True)
rare_action_types = data_cl['action_type'].value_counts().sort_values().index.values[:20]

data_cl.loc[data_cl['action_type'].isin(rare_action_types), 'action_type'] = 'Other'
categorial_cols = [

    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',

    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',

    'game_month', 'opponent']



for cc in categorial_cols:

    dummies = pd.get_dummies(data_cl[cc])

    dummies = dummies.add_prefix("{}_".format(cc))

    data_cl.drop(cc, axis=1, inplace=True)

    data_cl = data_cl.join(dummies)
data_submit = data_cl[unknown_mask]

# Separate dataset for training

X = data_cl[~unknown_mask]

Y = target[~unknown_mask]
d_train = xgb.DMatrix(X, label=Y)

dtest = xgb.DMatrix(data_submit)
params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['max_depth'] = 7

params['silent'] = 1

params['colsample_bytree'] = 0.7

params['eta'] = 0.004

params['max_delta_step'] = 1

params['min_child_weight'] = 3
#cvp = xgb.cv(params, d_train, num_boost_round=100000, early_stopping_rounds=10, metrics=['logloss'], verbose_eval=1, stratified=True)

#print("Best is {0} with logloss {1}".format(np.argmin(cvp['test-logloss-mean'] + cvp['test-logloss-std']), np.min(cvp['test-logloss-mean'] + cvp['test-logloss-std']))) 
#print("Best is {0} with logloss {1}".format(np.argmin(cvp['test-logloss-mean'] + cvp['test-logloss-std']), np.min(cvp['test-logloss-mean'] + cvp['test-logloss-std']))) 
clf = xgb.train(params, d_train, num_boost_round=961)
preds = clf.predict(dtest)

submission = pd.DataFrame()

submission["shot_id"] = data_submit.index

submission["shot_made_flag"]= preds



submission.to_csv("sub_xgb.csv",index=False)