import pandas as pd
import numpy as np
from sklearn import *
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
def data_prep(market_train,news_train):
    market_train.time = market_train.time.dt.date
    news_train.time = news_train.time.dt.hour
    news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
    news_train.firstCreated = news_train.firstCreated.dt.date
    news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
    news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
    kcol = ['firstCreated', 'assetCodes']
    news_train = news_train.groupby(kcol, as_index=False).mean()
    market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    
    
    market_train = market_train.dropna(axis=0)
    
    return market_train
market_train = data_prep(market_train,news_train)
# The target is binary
up = market_train.returnsOpenNextMktres10 >= 0
fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
# We still need the returns for model tuning
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
print (X.shape[0], up.shape[0], r.shape[0])
X_train, X_test, up_train, up_test, r_train, r_test\
= model_selection.train_test_split(X, up, r, test_size=0.25, random_state=99)
from xgboost import XGBClassifier
import time
xgb_up = XGBClassifier(n_jobs=4,n_estimators=200,max_depth=8,eta=0.1)
t = time.time()
print('Fitting Up')
xgb_up.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}')
from sklearn.metrics import accuracy_score
accuracy_score(xgb_up.predict(X_test),up_test)
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    t = time.time()
    market_obs_df = data_prep(market_obs_df, news_obs_df)
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = xgb_up.predict_proba(X_live)
    prediction_time += time.time() -t
    
    t = time.time()
    confidence = 2* lp[:,1] -1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost import plot_importance
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(range(len(xgb_up.feature_importances_)), xgb_up.feature_importances_)
plt.xticks(range(len(xgb_up.feature_importances_)), fcol, rotation='vertical');
