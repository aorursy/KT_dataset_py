import numpy as np
import time
import gc
import json
import os
from datetime import datetime
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
sns.set_style("dark")
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     parse_dates=['date'],
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

train_df = load_df()
test_df = load_df("../input/test.csv")
train_df.columns
train_df['train_or_test'] = 'train'
test_df['train_or_test'] = 'test'
test_df['totals.transactionRevenue'] = np.nan
df = pd.concat([train_df, test_df], sort=False, ignore_index=True)
del train_df
del test_df
gc.collect()
df.shape
df.head(10)
df['year'] = df.date.dt.year
df['month'] = df.date.dt.month
df['dayofmonth'] = df.date.dt.day
df['dayofweek'] = df.date.dt.dayofweek
df['dayofyear'] = df.date.dt.dayofyear
df['weekofyear'] = df.date.dt.weekofyear
df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
df['quarter'] = df.date.dt.quarter
df['week_block_num'] = [int(x) for x in np.floor((df.date - pd.to_datetime('2012-12-31')).dt.days/7) + 1]
df['quarter_block_num'] = (df['year'] - 2013) * 4 + df['quarter']
df.describe(include="all")
dropcols = [c for c in df.columns if df[c].nunique(dropna=True)==1]
dropcols.remove('totals.bounces')
dropcols.remove('totals.newVisits')
print(dropcols)
df.drop(dropcols,axis=1,inplace=True,errors='ignore')
df['totals.bounces'].fillna(0,inplace=True)
df['totals.hits'].fillna(0,inplace=True)
df['totals.pageviews'].fillna(0,inplace=True)
df['totals.newVisits'].fillna(0,inplace=True)
df[['totals.bounces','totals.hits','totals.pageviews','totals.newVisits']] = df[['totals.bounces','totals.hits','totals.pageviews','totals.newVisits']].astype(np.int)
null_df = df.isnull().sum().reset_index()
null_df[0] = null_df[0] / df.shape[0]
null_df[null_df[0] > 0]
cols = ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.slot','trafficSource.adContent']
df[cols] = df[cols].fillna("No_Ad")
df['trafficSource.adwordsClickInfo.page'].fillna(0,inplace=True)
df['trafficSource.referralPath'].fillna("No_Path",inplace=True)
df['trafficSource.adContent'].fillna("No_Ad",inplace=True)
df.describe(include=["O"]).T
cat_many_label_cols = ["channelGrouping", "device.browser", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source"]

cat_few_label_cols = ["device.deviceCategory","trafficSource.adwordsClickInfo.adNetworkType",
                     "trafficSource.adwordsClickInfo.slot"]
for col in cat_many_label_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(df[col].values.astype('str')))
    df[col] = lbl.transform(list(df[col].values.astype('str')))
    
df = pd.get_dummies(df,columns=cat_few_label_cols)
df.shape
df["totals.transactionRevenue"].fillna(0,inplace=True)
df["totals.transactionRevenue"] = df["totals.transactionRevenue"].astype(np.float)
train_df = df[df.train_or_test=='train']
test_df = df[df.train_or_test=='test'].drop('totals.transactionRevenue',axis=1)
val_df = train_df[train_df['date']>datetime(2017,5,31)]
print(train_df.shape)
print(val_df.shape)
print(test_df.shape)
dropcols = ['fullVisitorId','sessionId','visitId']
train_x = train_df.drop(dropcols,axis=1)
test_x = test_df.drop(dropcols,axis=1)
dev_x = train_x[train_x['date']<=datetime(2017,5,31)]
val_x = train_x[train_x['date']>datetime(2017,5,31)]
dev_y = np.log1p(dev_x["totals.transactionRevenue"].values)
val_y = np.log1p(val_x["totals.transactionRevenue"].values)
dev_x.drop(["totals.transactionRevenue","date","train_or_test"],axis=1,inplace=True)
val_x.drop(["totals.transactionRevenue","date","train_or_test"],axis=1,inplace=True)
test_x.drop(["date","train_or_test"],axis=1,inplace=True)
lgb_params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 1024,
        'max_depth': 16,  
        'max_bin': 255,
        "min_child_samples" : 100,
        "learning_rate" : 0.005,
        'verbose': 0,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018
    }
dtrain = lgb.Dataset(dev_x, label=dev_y)
dvalid = lgb.Dataset(val_x, label=val_y)
evals_results = {}
print("Training the model...")

start = datetime.now()
lgb_model = lgb.train(lgb_params, 
                 dtrain, 
                 valid_sets=[dtrain, dvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 num_boost_round=1000,
                 early_stopping_rounds=70,
                 verbose_eval=50, 
                 feval=None)
print("Total time taken : ", datetime.now()-start)
pred_test_lgb = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
pred_val_lgb = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
from sklearn import metrics
pred_val_lgb[pred_val_lgb<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val_lgb)
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = val_x.columns
fold_importance_df["importance"] = lgb_model.feature_importance()
plt.figure(figsize=(18,20))
sns.barplot(x='importance',y='feature',data=fold_importance_df.sort_values(by="importance", ascending=False))
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values
sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test_lgb[pred_test_lgb<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test_lgb)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("baseline_lgb.csv", index=False)