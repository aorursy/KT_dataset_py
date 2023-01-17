import os
import json
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from sklearn import model_selection, preprocessing, metrics
import matplotlib.pylab as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
%matplotlib inline
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
train_df = load_df()
test_df = load_df("../input/test.csv")
train_df.shape,test_df.shape
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0).astype(float)
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].apply(lambda x: np.log(x) if x > 0 else x)

train_df.drop(['trafficSource.campaignCode'], axis =1,inplace = True)
train_df.shape,test_df.shape
trn_len = train_df.shape[0]
df = pd.concat([train_df, test_df])
columns = [col for col in train_df.columns if train_df[col].nunique() > 1]

df = df[columns]
df.shape

import datetime

df['date'] = df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['weekday'] = df['date'].dt.weekday
cat_cols = ["channelGrouping", "device.browser", "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source"]

num_cols = ["totals.hits", "totals.pageviews" ,"month" ,"dayofweek" ,"weekday"]
target = "totals.transactionRevenue"
for col in cat_cols:
    lbl = preprocessing.LabelEncoder()
    df[col] = lbl.fit_transform(list(df[col].values.astype('str')))
for col in num_cols:
    df[col] = df[col].astype(float)
train_df = df[:trn_len]
test_df = df[trn_len:]
train_df.shape,test_df.shape
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values
tra_df = train_df[train_df['date']<=datetime.date(2017,4,31)]
val_df = train_df[train_df['date']>datetime.date(2017,4,31)]

tra_y = tra_df["totals.transactionRevenue"].values
val_y = val_df["totals.transactionRevenue"].values

tra_X = tra_df[cat_cols + num_cols]
val_X = val_df[cat_cols + num_cols]
test_X = test_df[cat_cols + num_cols]
param = {'num_leaves':40,
         'min_data_in_leaf': 15, 
         'objective':'regression',
         'max_depth': 8,
         'learning_rate':0.02,
         "min_child_samples":10,
        # "boosting":"rf",
         "feature_fraction":0.8,
         "bagging_freq":3,
         "bagging_fraction":0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 1,
         "verbosity": -1}
def run_lgb(params ,train_X, train_y, val_X, val_y, test_X):
        
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model


pred_test, model = run_lgb(param,tra_X, tra_y, val_X, val_y, test_X)
sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("baseline_lgb.csv", index=False)
features = cat_cols+num_cols
feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = features
feature_importance_df["importance"] = model.feature_importance()
    
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:1000].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')