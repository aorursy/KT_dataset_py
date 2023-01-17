import numpy as np
import lightgbm as lgb
import pandas as pd
from kaggle.competitions import twosigmanews
import matplotlib.pyplot as plt
import random
from datetime import datetime, date
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df.copy(), news_train_df.copy()
def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_df = mis_impute(market_train_df)

def data_prep(market_train):
    market_train.time = market_train.time.dt.date
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    
    market_train = market_train.dropna(axis=0)
    
    return market_train

market_train = data_prep(market_train_df)

# check the shape
print(market_train.shape)
market_train = market_train.loc[market_train['time']>=date(2009, 1, 1)]
up = market_train.returnsOpenNextMktres10 >= 0
fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X = market_train[fcol].values
up = up.values
r = market_train.returnsOpenNextMktres10.values

# Scaling of X values
# It is good to keep these scaling values for later
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

# Sanity check
assert X.shape[0] == up.shape[0] == r.shape[0]

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time

X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=99)

train_data = lgb.Dataset(X_train, label=up_train.astype(int))
test_data = lgb.Dataset(X_test, label=up_test.astype(int))
'''
# use this section if you want to customize optimization

# define blackbox function
def f(x):
    print(x)
    params = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x[0],
        'num_leaves': x[1],
        'min_data_in_leaf': x[2],
        'num_iteration': x[3],
        'max_bin': x[4],
        'verbose': 1
    }
    
    gbm = lgb.train(params,
            train_data,
            num_boost_round=100,
            valid_sets=test_data,
            early_stopping_rounds=5)
            
    print(type(gbm.predict(X_test, num_iteration=gbm.best_iteration)[0]),type(up_test.astype(int)[0]))
    
    print('score: ', mean_squared_error(gbm.predict(X_test, num_iteration=gbm.best_iteration), up_test.astype(float)))
    
    return mean_squared_error(gbm.predict(X_test, num_iteration=gbm.best_iteration), up_test.astype(float))

# optimize params in these ranges
spaces = [
    (0.19, 0.20), #learning_rate
    (2450, 2600), #num_leaves
    (210, 230), #min_data_in_leaf
    (310, 330), #num_iteration
    (200, 220) #max_bin
    ]

# run optimization
from skopt import gp_minimize
res = gp_minimize(
    f, spaces,
    acq_func="EI",
    n_calls=10) # increase n_calls for more performance

# print tuned params
print(res.x)

# plot tuning process
from skopt.plots import plot_convergence
plot_convergence(res)
'''
# these are tuned params I found
x_1 = [0.19000424246380565, 2452, 212, 328, 202]
x_2 = [0.19016805202090095, 2583, 213, 312, 220]

params_1 = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x_1[0],
        'num_leaves': x_1[1],
        'min_data_in_leaf': x_1[2],
        'num_iteration': x_1[3],
        'max_bin': x_1[4],
        'verbose': 1
    }

params_2 = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x_2[0],
        'num_leaves': x_2[1],
        'min_data_in_leaf': x_2[2],
        'num_iteration': x_2[3],
        'max_bin': x_2[4],
        'verbose': 1
    }


gbm_1 = lgb.train(params_1,
        train_data,
        num_boost_round=100,
        valid_sets=test_data,
        early_stopping_rounds=5)
        
gbm_2 = lgb.train(params_2,
        train_data,
        num_boost_round=100,
        valid_sets=test_data,
        early_stopping_rounds=5)
        

#prediction
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if (n_days%50==0):
        print(n_days,end=' ')
    t = time.time()
    market_obs_df = data_prep(market_obs_df)
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
    prediction_time += time.time() -t
    
    t = time.time()

    confidence = lp
    confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())
    confidence = confidence * 2 - 1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    
env.write_submission_file()
sub  = pd.read_csv("submission.csv")
