# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import shap

from bayes_opt import BayesianOptimization

from sklearn.model_selection import train_test_split, KFold, GridSearchCV



import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.svm import LinearSVR

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))



sns.set(color_codes=True)



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')
train.shape
train.head()
train.describe()
train.isnull().sum()
train[train['winPlacePerc'].isnull() == True]
train.drop(train.index[2744604], inplace=True)
plt.figure(figsize=(20,20))

corr = train.corr()

sns.heatmap(corr, annot=True)
sns.distplot(train['winPlacePerc'])
train[train['matchId']=='a10357fd1a4a91'].shape
sns.distplot(train[train['matchId']=='a10357fd1a4a91']['winPlacePerc'], bins=5)
#plt.figure(figsize=(24,16))



sns.violinplot(train['winPlacePerc'])
plt.title('Match Duration Distribution (s)')

plt.figure(figsize=(16,16))

plt.hist(train['matchDuration'])
# Feature Engineering





# Distance and speed

train['total_distance'] = train['swimDistance'] + train['walkDistance'] + train['rideDistance']

train['avg_speed'] = train['total_distance'] / train['matchDuration']

train['avg_swim_speed'] = train['swimDistance'] / train['matchDuration']

train['avg_walk_speed'] = train['walkDistance'] / train['matchDuration']

train['avg_ride_speed'] = train['rideDistance'] / train['matchDuration']

# Kill rate feature engineering

train['streak_rate'] = train['killStreaks'] / train['kills']

train['kills_rate'] = train['kills'] / train['matchDuration']

train['knocked_kill'] = train['DBNOs'] / train['kills']

train['kill_per_heal'] = train['heals'] / train['kills']

train['kills_per_place'] = train['kills'] / train['killPlace']

train['damage_kill'] = train['damageDealt'] / train['kills']

train['damage_rate'] = train['damageDealt'] / train['matchDuration']

# Utilities items

train['heals_rate'] = train['heals'] / train['matchDuration']

train['boosts_rate'] = train['boosts'] / train['matchDuration']

train['utility_used'] = train['boosts'] + train['heals']

train['boosts_prop'] = train['boosts'] / train['utility_used']

train['heals_prop'] = train['heals'] / train['utility_used']

train['utility_rate'] = train['utility_used'] / train['matchDuration']
train.replace([np.inf, -np.inf], np.nan, inplace=True)

train.fillna(0, inplace=True)
train.describe()
plt.figure(figsize=(16,16))

plt.title('Distribution: Data for player per matches')

plt.xlabel('Number of players per match')

plt.hist(train['numGroups'])

plt.figure(figsize=(24,16))

sns.scatterplot(train[train['matchId']=='a10357fd1a4a91']['walkDistance'], train[train['matchId']=='a10357fd1a4a91']['winPlacePerc'])

plt.title('Walk Distance vs. Winning Percentile (Match)')
plt.figure(figsize=(24,16))

sns.scatterplot(train['total_distance'], train['winPlacePerc'])

plt.title('Total Distance vs. Winning Percentile')
plt.figure(figsize=(24,16))

sns.scatterplot(train['avg_speed'], train['winPlacePerc'])

plt.title('Average Speed vs. Winning Percentile')
plt.figure(figsize=(24,16))

plt.title('Kills vs Winning Percentile')

sns.scatterplot(train['kills'], train['winPlacePerc'])

match_types = list(train['matchType'].unique())
fig, axs = plt.subplots(4,4, figsize=(16,19))

plt.subplots_adjust(hspace = 0.7)



for i,t in enumerate(match_types):

    axs[i // 4][i % 4].hist(train[train['matchType'] == t]['winPlacePerc'].astype('float'))

    axs[i // 4][i % 4].set_title(t+' Win Percentile')

     



fig, axs = plt.subplots(4,4, figsize=(20,30))

plt.subplots_adjust(hspace = 0.5)



for i,t in enumerate(match_types):

    sns.scatterplot(train[train['matchType'] == t]['total_distance'],train[train['matchType'] == t]['winPlacePerc'], ax=axs[i // 4][i % 4])

    axs[i // 4][i % 4].set_title(t+': Total Distance vs Win Percentile')
fig, axs = plt.subplots(4,4, figsize=(20,30))

plt.subplots_adjust(hspace = 0.5)



for i,t in enumerate(match_types):

    sns.scatterplot(train[train['matchType'] == t]['weaponsAcquired'],train[train['matchType'] == t]['winPlacePerc'], ax=axs[i // 4][i % 4])

    axs[i // 4][i % 4].set_title(t+': Weapons Acquired vs Win Percentile')
train['matchType'].unique()
# Check for fpp



def oh_matchtype(x, mode_name):

    if len(mode_name) <= len(x):

        if mode_name == x[:len(mode_name)]:

            return 1

    return 0

    

        

match_type = ['crash','flare','duo', 'solo', 'squad' ,'normal-duo','normal-solo','normal-squad']



train['fps_mode'] = train['matchType'].apply(lambda x: 1 if 'fpp' in x else 0)



for i in match_type:

    train['matchtype_'+i] = train['matchType'].apply(oh_matchtype, args=(i,))
train.drop(['matchType'], inplace=True, axis=1)
train.describe()
from sklearn.preprocessing import StandardScaler



#scale_col = ['damageDealt','matchDuration','rankPoints','killPoints','winPoints','walkDistance',

#             'rideDistance','damage_kill','total_distance','longestKill']

#train[scale_col] = StandardScaler().fit_transform(train[scale_col])

scaler = StandardScaler()

scaler.fit(train.drop(['Id','groupId','matchId','winPlacePerc'],axis=1))

from sklearn.metrics import mean_squared_error

from math import sqrt



train_red = train.sample(frac=0.1, random_state=42).reset_index(drop=True)
train_x, test_x, train_y, test_y = train_test_split(train_red.drop(['Id','groupId','matchId','winPlacePerc'],axis=1),

                                                    train_red['winPlacePerc'], 

                                                    test_size=0.25)



scale_train_x, scale_test_x, scale_train_y, scale_test_y = train_test_split(

    scaler.transform(train_red.drop(['Id','groupId','matchId','winPlacePerc'],axis=1)),

                                                    train_red['winPlacePerc'], 

                                                    test_size=0.25)
dtrain = xgb.DMatrix(train_x, train_y)

dtest = xgb.DMatrix(test_x, test_y)
xgb_log_params = {'eta': 0.5,

              'objective': 'reg:logistic',

              'max_depth': 7,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'eval_metric': ['rmse','mae'],

              'seed': 11,

              'silent': True}
watchlist = [(dtrain, 'train'), (dtest,'test')]
xgb_log_model = xgb.train(params=xgb_log_params, dtrain=dtrain, evals=watchlist)
explainer = shap.TreeExplainer(xgb_log_model)

shap_values = explainer.shap_values(train_x)

shap.summary_plot(shap_values, train_x)
def xgb_evaluate(max_depth, gamma, colsample_bytree):

    params = {'eval_metric': 'rmse',

              'max_depth': int(max_depth),

              'subsample': 0.8,

              'eta': 0.1,

              'gamma': gamma,

              'colsample_bytree': colsample_bytree}

    # Used around 1000 boosting rounds in the full model

    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    

    

    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE

    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 9), 

                                             'gamma': (0, 1),

                                             'colsample_bytree': (0.3, 0.9)})

# Use the expected improvement acquisition function to handle negative numbers

# Optimally needs quite a few more initiation points and number of iterations

xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
for param in xgb_bo.max['params']:

    xgb_log_params[param] = int(xgb_bo.max['params'][param])

    

xgb_log_params['eta'] = 0.09

num_boost_round = 10000

early_stopping_rounds=100
xgb_log_model = xgb.train(params=xgb_log_params, 

                          dtrain=dtrain, 

                          evals=[watchlist[1]],

                         early_stopping_rounds=early_stopping_rounds,

                         num_boost_round = num_boost_round)
def crossfit(model, X, y,n_splits = 5):

    kf = KFold(n_splits = 5, random_state=42)

    for train_index, test_index in kf.split(X):

        model = model

        train_X, train_Y = X[train_index], Y[train_index]

        test_X, test_Y = X[test_index], Y[test_index]

        model.fit(train_X, train_Y) 

        return sqrt(mean_squared_error(model.predict(test_X), test_Y))

del dtrain

del dtest
tuned_parameters = [{'tol': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]



clf = GridSearchCV(LinearSVR(), tuned_parameters, cv=4, scoring='neg_mean_squared_error')

clf.fit(scale_train_x, scale_train_y)

print(clf.best_params_)

means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("{} (+/-{}) for {}".format(mean, std * 2, params))
svr  = LinearSVR(max_iter=10000)

svr.set_params(**clf.best_params_)

svr_score = cross_val_score(svr, 

                scale_train_x, 

                scale_train_y, 

                cv=5,

                scoring='neg_mean_squared_error')

print(np.mean(np.sqrt(np.negative(svr_score))))