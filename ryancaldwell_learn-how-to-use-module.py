import os
import numpy as np
import pandas as pd
from datetime import timedelta
from xgboost import XGBClassifier
from kaggle.competitions import twosigmanews
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
def make_binary(field):
    return 1 if field > 0 else 0

def min_max(field):
    min_ = np.min(field)
    max_ = np.max(field)
    return [(x - min_)/(max_ - min_) for x in field]

def date_diff(dates):
    date_diffs = []
    
    last_date = dates[0]
    for d in dates:
        diff = (d - last_date)/timedelta(days=1)
        date_diffs.append(diff)
        last_date = d
    return date_diffs

def moving_average(field, run_length=10):
    means = []
    for i,v in enumerate(field):
        start = i-run_length
        if start < 0:
            start = 0
        m = np.mean(field[start:i])
        if np.isnan(m) == True:
            means.append(0.0)
        else:
            means.append(m)
    return means

def previous_changes(opens, closes, run_length=10):
    o = opens
    c = closes
    diff = []
    for i,v in enumerate(zip(o,c)):
        if i < run_length:
            diff.append(0.0)
        else:
            diff.append(v[1] - o[i - run_length])
            
    return [1 if x > 0.0 else 0 for x in diff]

def market_data_prep(market_df, train=True):
    market_train_df = market_df
    list_features = ['time','volume','close','open']
    if train:
        market_train_df['dep'] = market_train_df['returnsOpenNextMktres10'].apply(lambda x: make_binary(x))
        list_features.append('dep')

# 'time':lambda x: list(x),
# 'volume':lambda x: list(x),
# 'close':lambda x: list(x),
# 'open':lambda x: list(x),
# 'dep':lambda x: list(x)
        
    grouped_data = market_train_df.sort_values('time', ascending=True)\
                                  .groupby('assetCode')\
                                  .agg({'{}'.format(i):lambda x: list(x) for i in list_features})\
                                  .reset_index()
    print('Created our grouped lists.')
    
    grouped_data['prev_10'] = grouped_data[['open','close']].apply(lambda x: previous_changes(x['open'],x['close']), axis=1)
    print('Created previous 10 days movement')

    min_max_fields = ['volume','close','open']
    for f in min_max_fields:
        grouped_data[f] = grouped_data[f].apply(lambda x: min_max(x))
    print('Min Max Scaled our data')

    avg_fields = ['volume','close','open','prev_10']
    avg_len = [2,5,10]
    for f in avg_fields:
        for l in avg_len:
            grouped_data['{}_avg{}'.format(f,l)] = grouped_data[f].apply(lambda x: moving_average(x, run_length=l))
    print('Created our moving average features')

    grouped_data['date_diffs'] = grouped_data['time'].apply(lambda x: date_diff(x))

    grouped_data.drop('time', axis=1, inplace=True)

    grouped_data = grouped_data.set_index('assetCode')\
                               .apply(lambda x: x.apply(pd.Series).stack())\
                               .reset_index(level=1, drop=True)\
                               .reset_index()
    print('Finalized moving average features data frame')
    return grouped_data
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head(2)
del news_train_df
%%time
grouped_df = market_data_prep(market_train_df, train=True)
del market_train_df
grouped_df.shape
X_train, X_test, y_train, y_test = train_test_split(grouped_df.drop(['assetCode','dep'], axis=1), grouped_df.dep, 
                                                    test_size=0.20, 
                                                    random_state=42)
del grouped_df
%%time
xgb = XGBClassifier(n_jobs=4)
xgb.fit(X_train, y_train)
train_preds = xgb.predict_proba(X_train)[:,1]
test_preds = xgb.predict_proba(X_test)[:,1]

train_labels = xgb.predict(X_train)
test_labels = xgb.predict(X_test)
print(average_precision_score(y_train, train_preds))
print(roc_auc_score(y_train, train_preds))
print(accuracy_score(y_train, train_labels))
print(average_precision_score(y_test, test_preds))
print(roc_auc_score(y_test, test_preds))
print(accuracy_score(y_test, test_labels))
sorted(zip(xgb.feature_importances_, X_train.columns), reverse=True)
days = env.get_prediction_days()
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    market_df = market_data_prep(market_obs_df, train=False)
    model_preds = xgb.predict_proba(market_df.drop('assetCode', axis=1))[:,1]
    predictions_template_df['confidenceValue'] = [(2*x - 1.0) for x in model_preds]
    env.predict(predictions_template_df)
    
env.write_submission_file()
