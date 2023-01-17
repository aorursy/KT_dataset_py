# inspired from https://www.kaggle.com/jannesklaas/lb-0-63-xgboost-baseline
# https://www.kaggle.com/rabaman/0-64-in-100-lines
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market, news) = env.get_training_data()
(market.shape, news.shape)
market.head()
# asset code is enough, so remove asset name
market.drop(columns=['assetName'], inplace=True)
# change the time into just date.
market['time'] = market['time'].dt.strftime("%Y%m%d").astype(int)
market[(market['assetCode']=='FLEX.O') & (market['time'] > 20160701)].head(5)
market['trend'] = market['close'] / market['open'] - 1.
market.sort_values(by=['trend']).iloc[np.r_[0:5, -5:0]]
# Stock data with absurd (close/open) ratio are probably data errors, or at least outliers which are bad for learning. So let's remove them.
market = market[(market['trend'] < 1) & (market['trend'] > -0.8)]

# Also remove outliers in other columns
temp_cols = ['returnsOpenPrevRaw1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw1', 'returnsClosePrevMktres1', 
             'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10', 'returnsClosePrevRaw10', 'returnsClosePrevMktres10', 
             'returnsOpenNextMktres10']
for col in temp_cols:
    market = market[(1+market[col] > 0.5) & (1+market[col] < 2)]

market.shape
# Also, some of the Market-residualized numbers are dubious. Look at returnsClosePrevMktres1 on 2016-07-28 and 29 of Zinga.
market[(market['assetCode'] == 'ZNGA.O') & (market['time'] > 20160725 )].head(6)
# if the market-residualized value is too different from the raw value, remove it.

raw_cols = ['returnsOpenPrevRaw1', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw10', 'returnsClosePrevRaw10']
mktres_cols = ['returnsOpenPrevMktres1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres10', 'returnsClosePrevMktres10']

for (raw, mktres) in zip(raw_cols, mktres_cols):
    market = market[np.abs( (1+market[mktres])/(1+market[raw]) -1  ) < 0.2]
    
market.shape
# Rather than the absolute trading volume, volume ratio compared to the mean of each stock is probably more meaningful

# save the volume mean for prediction time
vol_mean = market.groupby('assetCode')['volume'].mean()
vol_mean = vol_mean.to_dict()

def compute_relvol(row):
    if row['assetCode'] in vol_mean:
        row['relvol'] = row['volume'] / vol_mean[row['assetCode']] - 1.
    else:
        row['relvol'] = 0.
    return row

# use groupby transform here since it's faster....
market['relvol'] = market.groupby('assetCode')['volume'].transform(lambda x: x / x.mean() - 1.)
market.head()
news.head()
# too many sources. probably not meaningful in learning
news['sourceId'].unique().shape
# Similarly for the providers
news['provider'].unique()
# headlines are broken string - do not really include a complete sentence or meaning
news.head()['headline'][3]
(news['marketCommentary'] == True).sum()
# treat firstMentionSentence == 0 as last sentence
news.loc[news['firstMentionSentence'] == 0, 'firstMentionSentence'] = news.loc[news['firstMentionSentence'] == 0, 'sentenceCount']
news['position'] = news['firstMentionSentence'] / news['sentenceCount']
news['coverage'] = news['sentimentWordCount'] / news['wordCount']
remove_news_cols = ['sourceTimestamp', 'firstCreated', 'sourceId', 'headline', 
               'provider', 'subjects', 'audiences', 'headlineTag', 'marketCommentary', 'assetName',
                'urgency', 'takeSequence', 'bodySize', 'companyCount', 'sentenceCount', 'wordCount', 
                'firstMentionSentence', 'sentimentClass', 'sentimentWordCount']
news.drop(columns=remove_news_cols, inplace=True)
# change the time into just date.
news['time'] = news['time'].dt.strftime("%Y%m%d").astype(int)
assetList = set(market['assetCode'].unique())
# majority of the rows include only one asset code
news['assetCodes'].map(lambda x: len(set(eval(x)) & assetList)).value_counts()
def codes2code(x):
    s = list(eval(x))
    for c in s:
        if c in assetList:
            return c    
    return np.nan
news['assetCode'] = news['assetCodes'].map(codes2code)
news.dropna(inplace=True)
news.drop(columns=['assetCodes'], inplace=True)
news.shape
# Note there are multiple news for the same day and same stock
news[news['assetCode']=='GOOG.O'].head(10)
# We will use relevance-weighted mean for those multiple news in a single day

for col in news.columns:
    if col not in ['time', 'relevance', 'position', 'assetCode']:
        news[col] = news[col] * news['relevance']

news = news.groupby(['time', 'assetCode'], as_index=False).mean()
news.drop(columns=['relevance'], inplace=True)
# now merge the market and news data
fulldata = pd.merge(market, news, how='inner', left_on=['time', 'assetCode'], 
                            right_on=['time', 'assetCode'])

for col in news.columns:
    if col not in ['time', 'position', 'assetCode']:
        fulldata[col].fillna(0, inplace=True)
    if col == 'position':
        fulldata[col].fillna(1, inplace=True)

(market.shape[0], news.shape[0], fulldata.shape[0])
fulldata.head(10)
fulldata.quantile([0.001, 0.2, 0.5, 0.8, 0.999])
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

fulldata.plot.hexbin(x='relvol', y='returnsOpenNextMktres10', gridsize=20, ax=ax[0][0], sharey=True, colorbar=False)
fulldata.plot.hexbin(x='trend', y='returnsOpenNextMktres10', gridsize=20, ax=ax[0][1], sharey=True, colorbar=False)
fulldata.plot.hexbin(x='returnsOpenPrevMktres10', y='returnsOpenNextMktres10', gridsize=20, ax=ax[0][2], sharey=True, colorbar=False)
fulldata.plot.hexbin(x='coverage', y='returnsOpenNextMktres10', gridsize=20, ax=ax[1][0], sharey=True, colorbar=False)
fulldata.plot.hexbin(x='volumeCounts24H', y='returnsOpenNextMktres10', gridsize=20, ax=ax[1][1], sharey=True, colorbar=False)
fulldata.plot.hexbin(x='sentimentPositive', y='returnsOpenNextMktres10', gridsize=20, ax=ax[1][2], sharey=True, colorbar=False)
fulldata.shape
outlier_threshold = 0.02
temp_cols = ['returnsClosePrevRaw1',
       'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
       'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
       'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
       'returnsOpenPrevMktres10', 'returnsOpenNextMktres10', 
       'trend', 'relvol']

temp = fulldata.loc[:, temp_cols]
data = fulldata[((temp < temp.quantile(1-outlier_threshold)) & (temp > temp.quantile(outlier_threshold))).all(axis=1)]
data.shape
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

data.plot.hexbin(x='relvol', y='returnsOpenNextMktres10', gridsize=20, ax=ax[0][0], sharey=True, colorbar=False)
data.plot.hexbin(x='trend', y='returnsOpenNextMktres10', gridsize=20, ax=ax[0][1], sharey=True, colorbar=False)
data.plot.hexbin(x='returnsOpenPrevMktres10', y='returnsOpenNextMktres10', gridsize=20, ax=ax[0][2], sharey=True, colorbar=False)
data.plot.hexbin(x='coverage', y='returnsOpenNextMktres10', gridsize=20, ax=ax[1][0], sharey=True, colorbar=False)
data.plot.hexbin(x='volumeCounts24H', y='returnsOpenNextMktres10', gridsize=20, ax=ax[1][1], sharey=True, colorbar=False)
data.plot.hexbin(x='sentimentPositive', y='returnsOpenNextMktres10', gridsize=20, ax=ax[1][2], sharey=True, colorbar=False)
xcol = [c for c in data.columns if c not in ['time', 'assetCode', 'assetName', 'close', 'open', 'volume',
                                             'returnsOpenNextMktres10', 'universe']]
dates = data['time'].unique()
train_days = data['time'].isin(dates[range(len(dates))[:int(0.85*len(dates))]])
test_days  = data['time'].isin(dates[range(len(dates))[int(0.85*len(dates)):]])

X_train = data[xcol].loc[train_days].values
X_test  = data[xcol].loc[test_days].values

y_train = (data.loc[train_days,'returnsOpenNextMktres10'] > 0).values
y_test  = (data.loc[test_days, 'returnsOpenNextMktres10'] > 0).values

r_train = data.loc[train_days,'returnsOpenNextMktres10'].values
r_test  = data.loc[test_days, 'returnsOpenNextMktres10'].values

u_train = data.loc[train_days,'universe'].values
u_test  = data.loc[test_days, 'universe'].values

d_train = data.loc[train_days,'time'].values
d_test  = data.loc[test_days, 'time'].values

(X_train.shape, X_test.shape)
from sklearn import *

def simple_metric(model):
    ret = metrics.accuracy_score(model.predict(X_test), y_test)
    print(ret)
    return ret

def real_metric(model, scale = 1.):
    # modified from https://www.kaggle.com/christofhenkel/market-data-nn-baseline
    preds = 2*model.predict_proba(X_test)[:,1]-1
    mean, std = np.mean(preds), np.std(preds)
    preds = (preds - mean)/ (std * scale)
    preds = np.clip(preds, -1, 1)
    
    x_t_i = preds * r_test * u_test
    df = pd.DataFrame({'day' : d_test, 'x_t_i' : x_t_i})
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    print(score_valid)
    return score_valid
# modified from http://blog.cypresspoint.com/2017/10/11/sklearn-random-forest-classification.html

import scipy.stats as st
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 100)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
class Random_model():
    def predict_proba(self, X):
        return np.random.random(size=(X.shape[0],2))
    def predict(self, X):
        return np.random.randint(2,size=X.shape[0])

simple_metric(Random_model())
real_metric(Random_model());
from lightgbm import LGBMClassifier

# money
params = {"objective" : "binary",
          "metric" : "binary_logloss",
          "num_leaves" : 60,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "verbosity" : -1 }


lgbm = LGBMClassifier(n_jobs=-1,silent=True, **params)
t = time.time()
print('lightgbm start')
lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=200)
print('Done, time = {}'.format(time.time() - t))

# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

# lgtrain, lgtest = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)
# lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgtest], early_stopping_rounds=100, verbose_eval=200)

model = lgbm
simple_metric(model)
real_metric(model, scale = 0.01)
real_metric(model, scale = 0.05)
real_metric(model, scale = 0.1)
real_metric(model, scale = 0.2)
real_metric(model, scale = 0.5)
real_metric(model, scale = 0.8)
real_metric(model, scale = 1)
real_metric(model, scale = 1.5)
real_metric(model, scale = 2)
real_metric(model, scale = 5)
real_metric(model, scale = 10)
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
to_plot = lgbm.feature_importances_
plt.bar(range(len(to_plot)), to_plot)
plt.xticks(range(len(to_plot)), xcol, rotation='vertical');
# from xgboost import XGBClassifier

# xgb_args = {'colsample_bytree': 0.9610500302744672,
#  'gamma': 0.6814215808960622,
#  'learning_rate': 0.43233807646582356,
#  'max_depth': 13,
#  'min_child_weight': 92.98682461481408,
#  'n_estimators': 72,
#  'reg_alpha': 165.46892179930694,
#  'subsample': 0.7079163067625419}

# xgb = XGBClassifier(n_jobs=-1,silent=True, **xgb_args)
# t = time.time()
# print('XGboost start')
# xgb.fit(X_train,y_train)
# print('Done, time = {}'.format(time.time() - t))
# model = xgb
# simple_metric(model)
# real_metric(model);
# plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
# to_plot = xgb.feature_importances_
# plt.bar(range(len(to_plot)), to_plot)
# plt.xticks(range(len(to_plot)), xcol, rotation='vertical');

# randomized search for hyperparameters
# # modified from http://danielhnyk.cz/how-to-use-xgboost-in-python/
# xgb_params = {  
#     "max_depth": st.randint(3, 40),
#     "learning_rate": st.uniform(0.05, 0.4),
#     "n_estimators": st.randint(10, 200),
#     "gamma": st.uniform(0, 10),
#     "min_child_weight": from_zero_positive,
#     "colsample_bytree": one_to_left,
#     "subsample": one_to_left,
#     'reg_alpha': from_zero_positive
# }

# xgbclass = XGBClassifier(n_jobs=-1,silent=True)
# xgb_random = RandomizedSearchCV(xgbclass, xgb_params, n_jobs=1, n_iter=10, cv=3, verbose=1)  

# t = time.time()
# print('XGboost start')
# xgb_random.fit(X_train,y_train)
# print('Done, time = {}'.format(time.time() - t))
# xgb_random.best_params_
# {'colsample_bytree': 0.9610500302744672,
#  'gamma': 0.6814215808960622,
#  'learning_rate': 0.43233807646582356,
#  'max_depth': 13,
#  'min_child_weight': 92.98682461481408,
#  'n_estimators': 72,
#  'reg_alpha': 165.46892179930694,
#  'subsample': 0.7079163067625419}
# xgb_random.best_score_
# model = xgb_random
# simple_metric(model)
# real_metric(model);



def data_prep(market, news):
    market.drop(columns=['assetName'], inplace=True)
    market['time'] = market['time'].dt.strftime("%Y%m%d").astype(int)
    market['trend'] = market['close'] / market['open'] - 1
    market = market.apply(compute_relvol, axis = 'columns')
    
    news.loc[news['firstMentionSentence'] == 0, 'firstMentionSentence'] = news.loc[news['firstMentionSentence'] == 0, 'sentenceCount']
    news['position'] = news['firstMentionSentence'] / news['sentenceCount']
    news['coverage'] = news['sentimentWordCount'] / news['wordCount']
    news.drop(columns=remove_news_cols, inplace=True)
    news['time'] = news['time'].dt.strftime("%Y%m%d").astype(int)
    news['assetCode'] = news['assetCodes'].map(codes2code)
    news.dropna(inplace=True)
    news.drop(columns=['assetCodes'], inplace=True)
    
    for col in news.columns:
        if col not in ['time', 'relevance', 'position', 'assetCode']:
            news[col] = news[col] * news['relevance']
    news = news.groupby(['time', 'assetCode'], as_index=False).mean()
    news.drop(columns=['relevance'], inplace=True)    
    
    data = pd.merge(market, news, how='left', left_on=['time', 'assetCode'], 
                            right_on=['time', 'assetCode'])
    for col in news.columns:
        if col not in ['time', 'position', 'assetCode']:
            data[col].fillna(0, inplace=True)
        if col == 'position':
            data[col].fillna(1, inplace=True)
            
    return data    
days = env.get_prediction_days()
model = lgbm
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs, news_obs, predictions_template) in days:
    n_days +=1
    if n_days % 30 == 0:
        print(n_days,end=' ')
    t = time.time()
    data_obs = data_prep(market_obs, news_obs)
    data_obs = data_obs[data_obs['assetCode'].isin(predictions_template['assetCode'])]
    X_live = data_obs[xcol].values
    prep_time += time.time() - t
    
    t = time.time()
    lp = model.predict_proba(X_live, num_iteration=model.best_iteration_)
    prediction_time += time.time() -t
    
    t = time.time()
    confidence = 2*lp[:,1] -1
    preds = pd.DataFrame({'assetCode':data_obs['assetCode'],'confidence':confidence})
    predictions_template = (predictions_template.merge(preds,how='left').drop('confidenceValue',axis=1)
                               .fillna(0).rename(columns={'confidence':'confidenceValue'}))
    env.predict(predictions_template)
    packaging_time += time.time() - t
env.write_submission_file()