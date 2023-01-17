import os
import numpy as np
import pandas as pd

import xgboost
import lightgbm as lgb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection
from datetime import datetime, date

%matplotlib inline
from kaggle.competitions import twosigmanews;
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# market_train_df = market_train_df[:2000000]
# news_train_df = news_train_df[:2000000]
maxs = None
mins = None
rng = None
mean_volume = None

means = None
stds = None
def drop_nans_and_infs(s:pd.Series):
    return s[~(np.isinf(s)|np.isnan(s))]
def nans_and_inf(X, val=None):
    """Replaces nans and infs with mean so that standart deviation normalization puts them to 0 """
    global means
    X = np.transpose(X)
    print(X.shape)
    if means is None:
        _means = np.mean(X, axis=0)
    else:
        _means = means
    print(_means)
    for i, m in zip(range(len(X)), _means):
        X[i][np.isnan(X[i])] = m
        X[i][np.abs(X[i]) > 1e7] = m
    X = np.transpose(X)
    return X
def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data
def _sd_norm(X):
    global means, stds
    if means is None or stds is None:
        means = np.mean(X)
        stds = np.std(X)
    return (X-means)/stds

def sd_norm(X):
    X = nans_and_inf(X)
    return _sd_norm(X)
def log_norm(X):
    X = nans_and_inf(X)
    X = np.log2(X)
    X = _sd_norm(X)
    return X
def min_max_norm(X):
    global maxs, mins, rng
    if maxs is None or mins is None or rng is None:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        rng = maxs - mins
    X = 1 - ((maxs - X) / rng)
    return X
def no_norm(X):
    return X

norm_method = min_max_norm
def remove_economic_crises(market_train):
    market_train = market_train.loc[market_train['time']>=date(2009, 1, 1)] # 2008 economic crises
#     market_train = market_train.loc[~((market_train['time']>=date(2011, 5, 2)) & (market_train['time']<=date(2011, 10, 4)))] # 2011
#     market_train = market_train.loc[~((market_train['time']>=date(2015, 6, 12)) & (market_train['time']<=date(2015, 8, 21)))] # 2015 Chinese and US market crashes
    return market_train
fcol = None
asset_code_map = None
headline_tag_map = None
provider_map = None
def add_hand_crafted_features(market_df):
    global mean_volume
    
    if mean_volume is None:
        mean_volume = market_df['volume'].mean()
    
    market_df['volume_to_mean'] = market_df['volume'] / mean_volume
    market_df['close_to_open_1'] = market_df['returnsClosePrevRaw1'] / market_df['returnsOpenPrevRaw1']
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df['returnsOpenPrevRaw1_to_volume'] = market_df['returnsOpenPrevRaw1'] / market_df['volume']
    market_df['returnsOpenPrevRaw10_to_volume'] = market_df['returnsOpenPrevRaw10'] / market_df['volume']
    return market_df

def data_prep(market_data):
    market_data = add_hand_crafted_features(market_data)
    market_data.time = market_data.time.dt.date
    global asset_code_map
    if asset_code_map is None:
        asset_code_map = {k: v for v, k in enumerate(market_data['assetCode'].unique())}
    market_data['assetCodeT'] = market_data['assetCode'].map(asset_code_map)
    
    market_data = market_data.dropna(axis=0)
    
    return market_data

def only_market(market_data):
    market_data = mis_impute(market_data)
    market_data = data_prep(market_data)
    return market_data

    
def market_and_news(market_data, news_data):
    market_data = mis_impute(market_data)
    news_data = mis_impute(news_data)
    
    market_data = add_hand_crafted_features(market_data)
    market_data.time = market_data.time.dt.date
    market_data = market_data.dropna(axis=0)
    market_data = remove_economic_crises(market_data)
    
    news_data['time'] = news_data.time.dt.hour
    news_data['sourceTimestamp']= news_data.sourceTimestamp.dt.hour
    news_data['firstCreated'] = news_data.firstCreated.dt.date
    news_data['assetCodesLen'] = news_data['assetCodes'].map(lambda x: len(eval(x)))
    news_data['assetCodes'] = news_data['assetCodes'].map(lambda x: list(eval(x))[0])
    news_data['headlineLen'] = news_data['headline'].apply(lambda x: len(x))
    news_data['assetCodesLen'] = news_data['assetCodes'].apply(lambda x: len(x))
    news_data['asset_sentiment_mean'] = news_data.groupby(['assetName', 'sentimentClass'])['time'].transform('mean')
    news_data['asset_sentiment_std'] = news_data.groupby(['assetName', 'sentimentClass'])['time'].transform('std')
    
    news_data['noveltyCount7D_mean'] = news_data.groupby(['assetName', 'noveltyCount7D'])['time'].transform('mean')
    news_data['noveltyCount7D_max'] = news_data.groupby(['assetName', 'noveltyCount7D'])['time'].transform('max')
    news_data['noveltyCount7D_min'] = news_data.groupby(['assetName', 'noveltyCount7D'])['time'].transform('min')
    
    news_data['asset_sentence_mean'] = news_data.groupby(['assetName', 'sentenceCount'])['time'].transform('mean')
    news_data['asset_sentence_max'] = news_data.groupby(['assetName', 'sentenceCount'])['time'].transform('max')
    news_data['asset_sentence_min'] = news_data.groupby(['assetName', 'sentenceCount'])['time'].transform('min')
    
    news_data['companyCount'] = news_data['companyCount'].astype(int)
    
    global headline_tag_map
    if headline_tag_map is None:
        headline_tag_map = {k: v for v, k in enumerate(news_data['headlineTag'].unique())}
    news_data['headlineTagT'] = news_data['headlineTag'].map(headline_tag_map)
    
    
    global provider_map
    if provider_map is None:
        provider_map = {k: v for v, k in enumerate(news_data['provider'].unique())}
    news_data['provider'] = news_data['provider'].map(provider_map)
    
    kcol = ['firstCreated', 'assetCodes']
    news_data = news_data.groupby(kcol, as_index=False).mean()
    
    market_data = pd.merge(market_data, news_data, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    
    global asset_code_map
    if asset_code_map is None:
        asset_code_map = {k: v for v, k in enumerate(market_data['assetCode'].unique())}
    market_data['assetCodeT'] = market_data['assetCode'].map(asset_code_map)
    
    market_data = mis_impute(market_data)
    return market_data
    

def train_processor(market_data, news_data=None):
    if news_data is None:
        market_data = only_market(market_data)
        market_data = remove_economic_crises(market_data)
    else:
        market_data = market_and_news(market_data, news_data)
    
    global fcol
    fcol = [c for c in market_data if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
    
    X = market_data[fcol].values
    up = market_data.returnsOpenNextMktres10 >= 0
    up = up.values
    r = market_data.returnsOpenNextMktres10.values

    # Scaling of X values
    # It is good to keep these scaling values for later
    X = norm_method(X)

    return X, up, r

def inferance_processor(market_data, predictions_template_df, news_data=None):
    if news_data is None:
        market_data = data_prep(market_data)
        market_data = market_data[market_data.assetCode.isin(predictions_template_df.assetCode)]
        X_live = market_data[fcol].values
        X_live = norm_method(X_live)
    else:
        market_data = market_and_news(market_data, news_data)
        market_data = market_data[market_data.assetCode.isin(predictions_template_df.assetCode)]
        X_live = market_data[fcol].values
        X_live = norm_method(X_live)
    return X_live, market_data


%%time
X, up, r = train_processor(market_train_df)
X.shape
%%time
X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.1, random_state=99)
xgb_train = xgboost.DMatrix(X_train, label=up_train, feature_names=fcol)

xgb_evals = xgboost.DMatrix(X_test, label=up_test, feature_names=fcol)
eval_list = [(xgb_train,'train'), (xgb_evals,'eval')]
params = {'eta': 0.15, 'max_depth': 6, 'max_bin': 300, 'booster': 'dart', 'objective': 'binary:logistic', 'eval_metric': ['auc', 'logloss'], 
          'is_training_metric': True, 'seed': 42, 'nthread': 4, 'gamma': 0.1, 'alpha': 0.1}
bst = xgboost.train(params, dtrain=xgb_train, num_boost_round= 500, evals=eval_list, early_stopping_rounds=30)
fig, ax = plt.subplots(1,1,figsize=[15,20])
xgboost.plot_importance(bst, ax=ax)
days = env.get_prediction_days()
import time

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if n_days % 50 == 0:
        print(n_days,end=' ')
    
    X_live, market_obs_df = inferance_processor(market_obs_df, predictions_template_df)
    d_live = xgboost.DMatrix(X_live, feature_names=fcol)
    
    t = time.time()
    lp = bst.predict(d_live)
    
    t = time.time()
    confidence = 2 * lp -1

    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    
env.write_submission_file()



