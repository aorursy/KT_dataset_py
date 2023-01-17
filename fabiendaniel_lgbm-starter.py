import pandas as pd
import numpy as np
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

print(os.listdir("../input"))
%%time
train_df = load_df()
test_df = load_df("../input/test.csv")
target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
target = target.apply(lambda x: np.log1p(x))
del train_df['totals.transactionRevenue']
columns_to_remove = [col for col in train_df.columns if train_df[col].nunique() == 1]
print("Nb. of variables with unique value: {}".format(len(columns_to_remove)))
for col in columns_to_remove:
    if set(['not available in demo dataset']) ==  set(train_df[col].unique()): continue
    print(col, train_df[col].dtypes, train_df[col].unique())
train_df['totals.bounces'] = train_df['totals.bounces'].fillna('0')
test_df['totals.bounces'] = test_df['totals.bounces'].fillna('0')

train_df['totals.newVisits'] = train_df['totals.newVisits'].fillna('0')
test_df['totals.newVisits'] = test_df['totals.newVisits'].fillna('0')

train_df['trafficSource.adwordsClickInfo.isVideoAd'] = train_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
test_df['trafficSource.adwordsClickInfo.isVideoAd'] = test_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)

train_df['trafficSource.isTrueDirect'] = train_df['trafficSource.isTrueDirect'].fillna(False)
test_df['trafficSource.isTrueDirect'] = test_df['trafficSource.isTrueDirect'].fillna(False)
columns = [col for col in train_df.columns if train_df[col].nunique() > 1]
#____________________________
train_df = train_df[columns]
test_df = test_df[columns]
trn_len = train_df.shape[0]
merged_df = pd.concat([train_df, test_df])
merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(int)
del merged_df['visitId']
del merged_df['sessionId']
format_str = '%Y%m%d' 
merged_df['formated_date'] = merged_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
merged_df['month'] = merged_df['formated_date'].apply(lambda x:x.month)
merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x:x.day//8)
merged_df['day'] = merged_df['formated_date'].apply(lambda x:x.day)
merged_df['weekday'] = merged_df['formated_date'].apply(lambda x:x.weekday())

del merged_df['date']
del merged_df['formated_date']
merged_df['totals.hits'] = merged_df['totals.hits'].astype(int)
merged_df['mean_hits_per_day'] = merged_df.groupby(['day'])['totals.hits'].transform('mean')
del  merged_df['day']
merged_df['formated_visitStartTime'] = merged_df['visitStartTime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
merged_df['formated_visitStartTime'] = pd.to_datetime(merged_df['formated_visitStartTime'])
merged_df['visit_hour'] = merged_df['formated_visitStartTime'].apply(lambda x: x.hour)

del merged_df['visitStartTime']
del merged_df['formated_visitStartTime']
# for col in ['totals.newVisits', 'totals.pageviews', 'totals.bounces']:
#     merged_df[col] = merged_df[col].astype(float)
# aggs = {
#         #'date': ['min', 'max'],
#         'totals.hits': ['sum', 'min', 'max', 'mean', 'median'],
#         'totals.pageviews': ['sum', 'min', 'max', 'mean'],
#         'totals.bounces': ['sum'],
#         'totals.newVisits': ['sum']
#     }
# users = merged_df.groupby('fullVisitorId').agg(aggs)
for col in merged_df.columns:
    if col in ['fullVisitorId', 'month', 'quarter_month', 'weekday', 'visit_hour', 'WoY']: continue
    if merged_df[col].dtypes == object or merged_df[col].dtypes == bool:
        merged_df[col], indexer = pd.factorize(merged_df[col])
numerics = [col for col in merged_df.columns if 'totals.' in col]
numerics += ['visitNumber', 'mean_hits_per_day', 'fullVisitorId']
categorical_feats =  [col for col in merged_df.columns if col not in numerics]
for col in categorical_feats:
    merged_df[col] = merged_df[col].astype(int)
#merged_df['fullVisitorId'] = merged_df['fullVisitorId'].astype(float)
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]
param = {'num_leaves': 300,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 1,
         "verbosity": -1}
trn_cols = [col for col in train_df.columns if col not in ['fullVisitorId']]
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
start = time.time()
features = list(train_df[trn_cols].columns)
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][trn_cols], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx][trn_cols], label=target.iloc[val_idx], categorical_feature=categorical_feats)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][trn_cols], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df[trn_cols], num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:1000].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = np.expm1(predictions)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test["PredictedLogRevenue"] = np.log1p(grouped_test["PredictedLogRevenue"])
grouped_test.to_csv('submit.csv',index=False)

