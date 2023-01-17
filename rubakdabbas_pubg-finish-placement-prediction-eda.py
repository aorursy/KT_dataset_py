# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PUBG_data = pd.read_csv('../input/train_V2.csv')
PUBG_data.head()
data_stat = PUBG_data.describe()
data_stat
data_stat.loc['count',:]!= 4.446966e+06
PUBG_data[PUBG_data['winPlacePerc'].isnull()]
PUBG_data = PUBG_data.drop(labels=2744604, axis=0)
PUBG_data[PUBG_data['winPlacePerc'].isnull()]
features_alive = ['assists', 'boosts', 'heals', 'revives']
len_f1 = len(features_alive)
fig = plt.figure()
fig.subplots_adjust(hspace=0.6, wspace=0.4)
for i in range(1, len_f1+1):
    ax = fig.add_subplot(2, 2, i)
    ax.scatter(PUBG_data[features_alive[i-1]], PUBG_data['winPlacePerc'])
    ax.set_xlabel(features_alive[i-1])
    ax.set_ylabel('winPlacePerc')
    corr_f = PUBG_data[[features_alive[i-1],'winPlacePerc']].corr().loc[features_alive[i-1],'winPlacePerc']
    ax.set_title('R^2= {:0.3f}'.format(corr_f))
def scatter_plot(x, y, x_label, y_label):
    plt.figure(figsize=(9,7))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
scatter_plot(PUBG_data['boosts'], PUBG_data['matchDuration']/60.0, 'boosts', 'match Duration (min)')
scatter_plot(PUBG_data['heals'], PUBG_data['matchDuration']/60.0, 'heals', 'match Duration (min)')
fig = plt.figure(figsize=(5,6))
fig.subplots_adjust(hspace=0.9, wspace=0.9)
for i in range(1, len_f1+1):
    ax = fig.add_subplot(2, 2, i)
    ax.scatter(PUBG_data[features_alive[i-1]]/(PUBG_data['matchDuration']/60.0), PUBG_data['winPlacePerc'])
    ax.set_xlabel(features_alive[i-1]+' '+'per min')
    ax.set_ylabel('winPlacePerc')
features_kill = ['headshotKills', 'killStreaks', 'kills', 'longestKill', 'roadKills', 
                 'vehicleDestroys']
len_f2 = len(features_kill)
fig = plt.figure()
fig.subplots_adjust(hspace=0.9, wspace=0.9)
for i in range(1, len_f2+1):
    ax = fig.add_subplot(2, 3, i)
    ax.scatter(PUBG_data[features_kill[i-1]], PUBG_data['winPlacePerc'])
    ax.set_xlabel(features_kill[i-1])
    ax.set_ylabel('winPlacePerc')
    corr_f = PUBG_data[[features_kill[i-1],'winPlacePerc']].corr().loc[features_kill[i-1],'winPlacePerc']
    ax.set_title('R^2= {:0.3f}'.format(corr_f))
scatter_plot(PUBG_data['kills'], PUBG_data['matchDuration']/60.0, 'kills', 'match Duration (min)')
scatter_plot(PUBG_data['headshotKills'], PUBG_data['matchDuration']/60.0, 'headshotKills', 'match Duration (min)')
scatter_plot(PUBG_data['killStreaks'], PUBG_data['matchDuration']/60.0, 'killStreaks', 'match Duration (min)')
scatter_plot(PUBG_data['longestKill'], PUBG_data['matchDuration']/60.0, 'longestKill', 'match Duration (min)')
features_kill2 = ['headshotKills', 'killStreaks', 'kills', 'longestKill', 'vehicleDestroys', 'roadKills']
len_f21 = len(features_kill2)
fig = plt.figure()
fig.subplots_adjust(hspace=0.9, wspace=0.9)
for i in range(1, len_f21+1):
    ax = fig.add_subplot(2, 3, i)
    ax.scatter(PUBG_data[features_kill2[i-1]]/(PUBG_data['matchDuration']/60.0), PUBG_data['winPlacePerc'])
    ax.set_xlabel(features_kill2[i-1]+' '+'per min')
    ax.set_ylabel('winPlacePerc')
features_distance = ['rideDistance', 'swimDistance', 'walkDistance']
len_f3 = len(features_distance)
fig = plt.figure()
fig.subplots_adjust(hspace=0.9, wspace=0.9)
for i in range(1, len_f3+1):
    ax = fig.add_subplot(1, 3, i)
    ax.scatter(PUBG_data[features_distance[i-1]]/1000, PUBG_data['winPlacePerc'])
    ax.set_xlabel(features_distance[i-1]+' '+'per km')
    ax.set_ylabel('winPlacePerc')
    corr_f = PUBG_data[[features_distance[i-1],'winPlacePerc']].corr().loc[features_distance[i-1],'winPlacePerc']
    ax.set_title('R^2= {:0.3f}'.format(corr_f))
features_points = ['killPoints', 'rankPoints', 'winPoints', 'maxPlace', 'killPlace']
len_f4 = len(features_points)
fig = plt.figure()
fig.subplots_adjust(hspace=0.9, wspace=0.9)
for i in range(1, len_f4+1):
    ax = fig.add_subplot(2, 3, i)
    ax.scatter(PUBG_data[features_points[i-1]], PUBG_data['winPlacePerc'])
    ax.set_xlabel(features_points[i-1])
    ax.set_ylabel('winPlacePerc')
    corr_f = PUBG_data[[features_points[i-1],'winPlacePerc']].corr().loc[features_points[i-1],'winPlacePerc']
    ax.set_title('R^2= {:0.3f}'.format(corr_f))
scatter_plot(PUBG_data['killPoints']/PUBG_data['maxPlace'], PUBG_data['winPlacePerc'], 
             'killPoints per maxPlace ', 'winPlacePerc')
scatter_plot(PUBG_data['winPoints']/PUBG_data['maxPlace'], PUBG_data['winPlacePerc'], 
             'winPoints per maxPlace ', 'winPlacePerc')
scatter_plot(PUBG_data['rankPoints']/PUBG_data['maxPlace'], PUBG_data['winPlacePerc'], 
             'rankPoints per maxPlace', 'winPlacePerc')
scatter_plot(PUBG_data['killPlace']/PUBG_data['maxPlace'], PUBG_data['winPlacePerc'], 
             'killPlace per maxPlace', 'winPlacePerc')
players_in_team = PUBG_data.groupby(['groupId']).size().to_frame('players_in_team').reset_index(level='groupId')
PUBG_data= PUBG_data.merge(players_in_team, on='groupId')
num_match = PUBG_data.groupby(['matchId']).size().to_frame('num_match').reset_index(level='matchId')
num_match.head()
PUBG_data= PUBG_data.merge(num_match, on='matchId')
PUBG_data.head()
data_stat2 = PUBG_data.describe()
data_stat2
data_stat2.loc['count',:]!= 4.446965e+06
def player_match_num(data):
    players_in_team = data.groupby(['groupId']).size().to_frame('players_in_team').reset_index(level='groupId')
    num_match = data.groupby(['matchId']).size().to_frame('num_match').reset_index(level='matchId')
    data= data.merge(players_in_team, on='groupId')
    data= data.merge(num_match, on='matchId')
    return data    
def feature_alive(data):
    features = ['assists', 'boosts', 'heals', 'revives']
    new_data = pd.DataFrame()
    for i in features:
        new_data[i +' '+'per min'] = data[i]/(data['matchDuration']/60.0)
    return new_data
features_alive = feature_alive(PUBG_data)
def feature_kill(data):
    features = ['headshotKills', 'killStreaks', 'kills', 'vehicleDestroys', 'killPlace']
    new_data = pd.DataFrame()
    for i in features:
        new_data[i +' '+'per min'] = data[i]/(data['matchDuration']/60.0)
    return new_data
features_kill = feature_kill(PUBG_data)
def feature_distance(data):
    features = ['rideDistance', 'swimDistance', 'walkDistance']
    new_data = pd.DataFrame()
    for i in features:
        new_data[i +' '+'km'] = data[i]/1000
    return new_data
features_distance = feature_distance(PUBG_data)
def feature_points(data):
    features = ['killPoints', 'rankPoints', 'winPoints']
    new_data = pd.DataFrame()
    for i in features:
        new_data[i +' / '+'maxPlace'] = data[i]/data['maxPlace']
    return new_data
features_points = feature_points(PUBG_data)
def dummy_feature(data):
    new_data = pd.get_dummies(data['matchType'])
    return new_data
dummy_features = dummy_feature(PUBG_data)
features1 = ['assists', 'boosts', 'heals', 'revives']
features2 = ['headshotKills', 'killStreaks', 'kills', 'vehicleDestroys', 'roadKills', 'matchDuration', 'killPlace']
features3 = ['rideDistance', 'swimDistance', 'walkDistance']
features4 = ['killPoints', 'rankPoints', 'winPoints', 'maxPlace']
features5 = ['matchType', 'Id', 'groupId', 'matchId', 'winPlacePerc']
PUBG_features = PUBG_data.copy()
y = PUBG_data['winPlacePerc']
X = PUBG_features.drop(features1+features2+features3+features4+features5, axis=1)
list1 = features_alive.columns
list2 = features_kill.columns
list3 = features_distance.columns
list4 = features_points.columns
list5 = dummy_features.columns
X[list1] = features_alive
X[list2] = features_kill
X[list3] = features_distance
X[list4] = features_points
X[list5] = dummy_features
PUBG_test = pd.read_csv('../input/test_V2.csv')
PUBG_test = player_match_num(PUBG_test)
data_stat_test = PUBG_test.describe()
data_stat_test
features51 = ['matchType', 'Id', 'groupId', 'matchId']
X_test = PUBG_test.drop(features1+features2+features3+features4+features51, axis=1)
X_test[list1] = feature_alive(PUBG_test)
X_test[list2] = feature_kill(PUBG_test)
X_test[list3] = feature_distance(PUBG_test)
X_test[list4] = feature_points(PUBG_test)
X_test[list5] = dummy_feature(PUBG_test)
X_test.head()
from sklearn.model_selection import train_test_split

def train_valid(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.4)
    return train_X, val_X, train_y, val_y
from sklearn.metrics import mean_absolute_error
def mae(y_predictions, y):
    val_mae = mean_absolute_error(y_predictions, y)
    return val_mae
train_X, val_X, train_y, val_y = train_valid(X, y)
from lightgbm import LGBMRegressor
lbgm = LGBMRegressor(boosting_type='gbdt', importance_type='split', learning_rate=0.45,
                     max_depth=-10, n_estimators= 1200, num_leaves=31, 
                     objective='regression_l2', reg_alpha=0.01, reg_lambda=100)
lbgm.fit(train_X, train_y, eval_set=[(val_X, val_y)], eval_metric ='mae', verbose=100)
feature_importance = pd.DataFrame()
feature_importance["feature"] = train_X.columns
feature_importance["importance"] = lbgm.feature_importances_
feature_importance.sort_values('importance', ascending = False).reset_index().drop('index', axis = 1)
def pred(X, model):
    y_pred =  model.predict(X)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 1] = 1
    return y_pred
y_pred_train =  pred(train_X, lbgm)
print('MAE_train = ', mae(y_pred_train, train_y))
y_pred = pred(val_X, lbgm)
print('MAE_validation = ', mae(y_pred, val_y))
y_pred_test = pred(X_test, lbgm)
sub_data = pd.DataFrame({'Id':PUBG_test['Id'],'winPlacePerc': y_pred_test})
sub_data.to_csv('submission.csv', index=False)