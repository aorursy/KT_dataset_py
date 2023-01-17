#import
from kaggle.competitions import twosigmanews
from datetime import datetime, date
import numpy as np
from sklearn import model_selection
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df.copy(), news_train_df.copy()
def fill_in(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_df = fill_in(market_train_df)
def data_other(market_train_df):
    market_train_df.time = market_train_df.time.dt.date
    lbl = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
    market_train_df['assetCodeT'] = market_train_df['assetCode'].map(lbl)
    
    market_train_df = market_train_df.dropna(axis=0)
    
    return market_train_df

market_train_df = data_other(market_train_df)
market_train_df = market_train_df.loc[market_train_df['time']>=date(2009, 1, 1)]
green = market_train_df.returnsOpenNextMktres10 > 0
green = green.values
fcol = [c for c in market_train_df if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
X = market_train_df[fcol].values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
all_10day = market_train_df.returnsOpenNextMktres10.values
X_train, X_test, green_train, green_test, all_train, all_test = model_selection.train_test_split(
    X, green, all_10day, test_size=0.20, random_state=59)
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=green_train.astype(int))
test_data = lgb.Dataset(X_test, label=green_test.astype(int))
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
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
import pandas as pd
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if (n_days%50==0):
        print(n_days,end=' ')
    market_obs_df = fill_in(market_obs_df)
    market_obs_df = data_other(market_obs_df)
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    
    lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
    

    confidence = lp
    confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())
    confidence = confidence * 2 - 1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
env.write_submission_file()
