import pandas as pd
import re
import nltk
import keras
import tensorflow as tf
import gensim
import numpy as np
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
import math
from sklearn import metrics
import matplotlib.pyplot as plt
#####################################################3
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
data = [go.Heatmap(
        z= train.corr().values,
        x=train.columns.values,
        y=train.columns.values,
        colorscale='Viridis',
        reversescale = False,
        #text = True,
        opacity = 1.0)]

layout = go.Layout(
    title='Pearson Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
margin=dict(
    l=240,
),)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
train['assists_perf'] = train.groupby(['matchId','groupId']).assists.transform('sum')/train.groupby(['matchId','groupId']).Id.transform('count')
train['assists_perf'] = train['assists_perf']/train.groupby('matchId').assists_perf.transform('max')
train['revives_perf'] = train.groupby(['matchId','groupId']).revives.transform('sum')/train.groupby(['matchId','groupId']).Id.transform('count')
train['revives_perf'] = train['revives_perf']/train.groupby('matchId').revives_perf.transform('max')
# Measurement on teamwork.
train['dmgdealt_perf'] = train.groupby(['matchId','groupId']).damageDealt.transform('sum')/train.groupby(['matchId','groupId']).Id.transform('count')
train['dmgdealt_perf'] = train['dmgdealt_perf']/train.groupby('matchId').dmgdealt_perf.transform('max')
# Measurement on damage.
test['assists_perf'] = test.groupby(['matchId','groupId']).assists.transform('sum')/test.groupby(['matchId','groupId']).Id.transform('count')
test['assists_perf'] = test['assists_perf']/test.groupby('matchId').assists_perf.transform('max')
test['revives_perf'] = test.groupby(['matchId','groupId']).revives.transform('sum')/test.groupby(['matchId','groupId']).Id.transform('count')
test['revives_perf'] = test['revives_perf']/test.groupby('matchId').revives_perf.transform('max')
test['dmgdealt_perf'] = test.groupby(['matchId','groupId']).damageDealt.transform('sum')/test.groupby(['matchId','groupId']).Id.transform('count')
test['dmgdealt_perf'] = test['dmgdealt_perf']/test.groupby('matchId').dmgdealt_perf.transform('max')
def modelfit(alg,dtrain,predictors,useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain['winPlacePerc'].values, feature_names = predictors)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds, metrics = 'mae', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators = cvresult.shape[0])
        print('Best n_estimator = ' + str(cvresult.shape[0]))
    alg.fit(dtrain[predictors], dtrain['winPlacePerc'], eval_metric = 'mae')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    print('\nModel Report:')
    print('MAE: %f' % math.sqrt(metrics.mean_absolute_error(dtrain['winPlacePerc'].values, dtrain_predictions)))
train.columns
ready_train = train[['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       'winPlacePerc', 'assists_perf', 'revives_perf', 'dmgdealt_perf']]
ready_test = test[['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints', 'assists_perf', 'revives_perf', 'dmgdealt_perf']]
predictors = ready_train.columns[ready_train.columns != 'winPlacePerc']
len(predictors)
xgb1 = XGBRegressor(objective = 'reg:logistic', learning_rate = 0.1, n_estimators = 50, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 1, seed = 2018)
modelfit(xgb1, ready_train, predictors, useTrainCV = False)
xgb.plot_importance(xgb1)
plt.show()