import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold

warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
%matplotlib inline
train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape
y_clf = (train['totals.transactionRevenue'].fillna(0) > 0).astype(np.uint8)
y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']
y_clf.mean(), y_reg.mean()
for df in [train, test]:
    df['date'] = pd.to_datetime(df['date'])
    df['vis_date'] = pd.to_datetime(df['visitStartTime'])
    df['sess_date_dow'] = df['vis_date'].dt.dayofweek
    df['sess_date_hours'] = df['vis_date'].dt.hour
    df['sess_date_dom'] = df['vis_date'].dt.day
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime', 'non_zero_proba', 'vis_date'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]

if 'totals.transactionRevenue' in train.columns:
    del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']
for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])
folds = GroupKFold(n_splits=5)

train_features = [_f for _f in train.columns if _f not in excluded_features]
print(train_features)
oof_clf_preds = np.zeros(train.shape[0])
sub_clf_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds.split(y_clf, y_clf, groups=train['fullVisitorId'])):
    trn_x, trn_y = train[train_features].iloc[trn_], y_clf.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_clf.iloc[val_]
    
    clf = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    clf.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=50,
        verbose=50
    )
    
    oof_clf_preds[val_] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    print(roc_auc_score(val_y, oof_clf_preds[val_]))
    sub_clf_preds += clf.predict_proba(test[train_features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
roc_auc_score(y_clf, oof_clf_preds)
train['non_zero_proba'] = oof_clf_preds
test['non_zero_proba'] = sub_clf_preds
train_features = [_f for _f in train.columns if _f not in excluded_features] + ['non_zero_proba']
print(train_features)

oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds.split(y_reg, y_reg, groups=train['fullVisitorId'])):
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_].fillna(0)
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_].fillna(0)
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=50
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / folds.n_splits
    
mean_squared_error(np.log1p(y_reg.fillna(0)), oof_reg_preds) ** .5
import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
test['PredictedLogRevenue'] = sub_reg_preds
test[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum()['PredictedLogRevenue'].apply(np.log1p).reset_index()\
    .to_csv('test_clf_reg_log_of_sum.csv', index=False)
# Go to actual revenues
train['PredictedRevenue'] = np.expm1(oof_reg_preds)
test['PredictedRevenue'] = sub_reg_preds
train['totals.transactionRevenue'] = y_reg

# Sum by date on train and test
trn_group = train[['date', 'PredictedRevenue', 'totals.transactionRevenue']].groupby('date').sum().reset_index()
sub_group = test[['date', 'PredictedRevenue']].groupby('date').sum().reset_index()

# Now plot all this
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y-%m')

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_title('Actual Dollar Revenues - we are way off...', fontsize=15, fontweight='bold')
ax.plot(pd.to_datetime(trn_group['date']).values, trn_group['totals.transactionRevenue'].values)
ax.plot(pd.to_datetime(trn_group['date']).values, trn_group['PredictedRevenue'].values)
ax.plot(pd.to_datetime(sub_group['date']).values, sub_group['PredictedRevenue'].values)

# # format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
# # ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
# Go to actual revenues
train['PredictedRevenue'] = np.expm1(oof_reg_preds)
test['PredictedRevenue'] = sub_reg_preds
train['totals.transactionRevenue'] = y_reg

# Sum by date on train and test
trn_group = train[['date', 'PredictedRevenue', 'totals.transactionRevenue']].groupby('date').sum().reset_index()
sub_group = test[['date', 'PredictedRevenue']].groupby('date').sum().reset_index()

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y-%m')

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_title('We are also off in logs... or am I just stupid ?', fontsize=15, fontweight='bold')
ax.plot(pd.to_datetime(trn_group['date']).values, np.log1p(trn_group['totals.transactionRevenue'].values))
ax.plot(pd.to_datetime(trn_group['date']).values, np.log1p(trn_group['PredictedRevenue'].values))
ax.plot(pd.to_datetime(sub_group['date']).values, np.log1p(sub_group['PredictedRevenue'].values))

# # format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
# # ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
# Keep amounts in logs
train['PredictedRevenue'] = oof_reg_preds
test['PredictedRevenue'] = np.log1p(sub_reg_preds)
train['totals.transactionRevenue'] = np.log1p(y_reg)

# You really mean summing up the logs ???
trn_group = train[['date', 'PredictedRevenue', 'totals.transactionRevenue']].groupby('date').sum().reset_index()
sub_group = test[['date', 'PredictedRevenue']].groupby('date').sum().reset_index()

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y-%m')

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_title('Summing up logs looks a lot better !?! Is the challenge to find the correct metric ???', fontsize=15, fontweight='bold')
ax.plot(pd.to_datetime(trn_group['date']).values, trn_group['totals.transactionRevenue'].values)
ax.plot(pd.to_datetime(trn_group['date']).values, trn_group['PredictedRevenue'].values)
ax.plot(pd.to_datetime(sub_group['date']).values, sub_group['PredictedRevenue'].values)

# # format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
# # ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()