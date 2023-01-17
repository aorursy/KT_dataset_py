import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell #To print multiple outputs

InteractiveShell.ast_node_interactivity = 'all'
train_df = pd.read_csv('../input/train.csv', skiprows=range(1,123903891), nrows = 1000000, parse_dates=['click_time'])

test_df = pd.read_csv('../input/test.csv', parse_dates=['click_time'], iterator=True)

sub_df = pd.read_csv('../input/sample_submission.csv')
test_df1 = test_df.get_chunk(10000000)

test_df2 = test_df.get_chunk(8790469)

test_df1.shape

test_df2.shape
train_df.head()
train_df.info()
train_df.shape

test_df1.shape

test_df2.shape

sub_df.shape
train_df = train_df.append(test_df)

train_df.shape
len(train_df.ip.unique()), len(train_df.app.unique()), len(train_df.channel.unique()), len(train_df.device.unique())
train_df['day_of_week'] = train_df['click_time'].dt.dayofweek

train_df['hour'] = train_df['click_time'].dt.hour

train_df['day'] = train_df['click_time'].dt.day

train_df['month'] = train_df['click_time'].dt.month

train_df['IsWeekend'] = train_df['day_of_week'].apply(lambda x : 0 if x==0 | x==6 else 1)
test_df1['day_of_week'] = test_df1['click_time'].dt.dayofweek

test_df1['hour'] = test_df1['click_time'].dt.hour

test_df1['day'] = test_df1['click_time'].dt.day

test_df1['month'] = test_df1['click_time'].dt.month

test_df1['IsWeekend'] = test_df1['day_of_week'].apply(lambda x : 0 if x==0 | x==6 else 1)
test_df2['day_of_week'] = test_df2['click_time'].dt.dayofweek

test_df2['hour'] = test_df2['click_time'].dt.hour

test_df2['day'] = test_df2['click_time'].dt.day

test_df2['month'] = test_df2['click_time'].dt.month

test_df2['IsWeekend'] = test_df2['day_of_week'].apply(lambda x : 0 if x==0 | x==6 else 1)
import time

train_df["ordinal_date"] = train_df["click_time"].apply(lambda x: time.mktime(x.timetuple()))

test_df1["ordinal_date"] = test_df1["click_time"].apply(lambda x: time.mktime(x.timetuple()))

test_df2["ordinal_date"] = test_df2["click_time"].apply(lambda x: time.mktime(x.timetuple()))

train_df.head()
train_df = train_df.sort_values(by="ordinal_date").reset_index(drop=True)

test_df1 = test_df1.sort_values(by="ordinal_date").reset_index(drop=True)

test_df2 = test_df2.sort_values(by="ordinal_date").reset_index(drop=True)

train_df.head()
min(train_df.click_time.dt.date)
max(train_df.click_time.dt.date)
click_counts_byappid_train = train_df.groupby(['ip','app'])['click_time'].size().rename('ipcount_byapp').reset_index()

click_counts_byappid_test1 = test_df1.groupby(['ip','app'])['click_time'].size().rename('ipcount_byapp').reset_index()

click_counts_byappid_test2 = test_df2.groupby(['ip','app'])['click_time'].size().rename('ipcount_byapp').reset_index()

click_counts_byappid_train.head()

click_counts_byappid_train.tail()
click_counts_byappid_train.shape
train_df_new = pd.merge(train_df, click_counts_byappid_train, on=['ip','app'], how='left', sort=False)

test_df_new1 = pd.merge(test_df1, click_counts_byappid_test1, on=['ip','app'], how='left', sort=False)

test_df_new2 = pd.merge(test_df2, click_counts_byappid_test2, on=['ip','app'], how='left', sort=False)
test_df_new1.shape

test_df_new2.shape
click_counts_bychannel_train = train_df.groupby(['ip','channel'])['click_time'].size().rename('ipcount_bychannel').reset_index()

click_counts_bychannel_test1 = test_df1.groupby(['ip','channel'])['click_time'].size().rename('ipcount_bychannel').reset_index()

click_counts_bychannel_test2 = test_df2.groupby(['ip','channel'])['click_time'].size().rename('ipcount_bychannel').reset_index()

click_counts_bychannel_train.shape

click_counts_bychannel_train.head()

click_counts_bychannel_train.tail()
train_df_new = pd.merge(train_df_new, click_counts_bychannel_train, on=['ip','channel'], how='left', sort=False)

test_df_new1 = pd.merge(test_df_new1, click_counts_bychannel_test1, on=['ip','channel'], how='left', sort=False)

test_df_new2 = pd.merge(test_df_new2, click_counts_bychannel_test2, on=['ip','channel'], how='left', sort=False)

train_df_new.head()
click_counts_bydevice_train = train_df.groupby(['ip','device'])['click_time'].size().rename('ipcount_bydevice').reset_index()

click_counts_bydevice_test1 = test_df1.groupby(['ip','device'])['click_time'].size().rename('ipcount_bydevice').reset_index()

click_counts_bydevice_test2 = test_df2.groupby(['ip','device'])['click_time'].size().rename('ipcount_bydevice').reset_index()

click_counts_bydevice_train.shape

click_counts_bydevice_train.head()

click_counts_bydevice_train.tail()
train_df_new = pd.merge(train_df_new, click_counts_bydevice_train, on=['ip','device'], how='left', sort=False)

test_df_new1 = pd.merge(test_df_new1, click_counts_bydevice_test1, on=['ip','device'], how='left', sort=False)

test_df_new2 = pd.merge(test_df_new2, click_counts_bydevice_test2, on=['ip','device'], how='left', sort=False)

train_df_new.head()
!ls -ltr
train_df_new["ip_cum_count"] = train_df_new.groupby("ip")["device"].cumcount()

test_df_new1["ip_cum_count"] = test_df_new1.groupby("ip")["device"].cumcount()

test_df_new2["ip_cum_count"] = test_df_new2.groupby("ip")["device"].cumcount()

train_df_new.head()
train_df_new["prev_date"] = train_df_new.groupby("ip")["ordinal_date"].shift(1)

test_df_new1["prev_date"] = test_df_new1.groupby("ip")["ordinal_date"].shift(1)

test_df_new2["prev_date"] = test_df_new2.groupby("ip")["ordinal_date"].shift(1)

train_df_new.head()

test_df_new1.head()
train_df_new["date_diff"] = train_df_new["ordinal_date"] - train_df_new["prev_date"]

test_df_new1["date_diff"] = test_df_new1["ordinal_date"] - test_df_new1["prev_date"]

test_df_new2["date_diff"] = test_df_new2["ordinal_date"] - test_df_new2["prev_date"]

train_df_new.head()

test_df_new1.head()
gdf = train_df_new.groupby("ip")["ordinal_date"].agg(["min", "mean", "max", "std"]).reset_index()

gdf1 = test_df_new1.groupby("ip")["ordinal_date"].agg(["min", "mean", "max", "std"]).reset_index()

gdf2 = test_df_new2.groupby("ip")["ordinal_date"].agg(["min", "mean", "max", "std"]).reset_index()

gdf.columns = ["ip", "min_date", "mean_date", "max_date", "std_date"]

gdf1.columns = ["ip", "min_date", "mean_date", "max_date", "std_date"]

gdf2.columns = ["ip", "min_date", "mean_date", "max_date", "std_date"]

train_df_new = pd.merge(train_df_new, gdf, on="ip")

test_df_new1 = pd.merge(test_df_new1, gdf1, on="ip")

test_df_new2 = pd.merge(test_df_new2, gdf2, on="ip")

train_df_new.head()

test_df_new1.head()
gdf = train_df_new[train_df_new['is_attributed']==1]

gdf = gdf.groupby("ip")["ordinal_date"].agg(["mean","std"]).reset_index()

gdf.columns = ["ip", "ip_mean_date_click", "ip_std_date_click"]

train_df_new = pd.merge(train_df_new, gdf, on="ip", how="left")

train_df_new.head()
test_df_new1 = pd.merge(test_df_new1, gdf, on="ip", how="left")

test_df_new2 = pd.merge(test_df_new2, gdf, on="ip", how="left")

test_df_new1.head()

test_df_new2.head()
train_df_new.to_csv('train_df_new.csv', index=False)
test = pd.concat([test_df_new1, test_df_new2])
test.to_csv('test_df.csv', index=False)
import pandas as pd

train_df = pd.read_csv('train_df_new.csv')

test_df1 = pd.read_csv('test_df_new1.csv')

test_df2 = pd.read_csv('test_df_new2.csv')
feature_cols = train_df.columns.drop(['ip','attributed_time', 'is_attributed','click_time','ordinal_date','prev_date'])

feature_cols
X = train_df[feature_cols]

y = train_df.is_attributed
from sklearn.model_selection import train_test_split    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 33)
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_test, y_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
params = {'eta': 0.1, # learning rate

          'tree_method': "auto", 

          'max_depth': 4, 

          'subsample': 0.8, 

          'colsample_bytree': 0.7, 

          'colsample_bylevel':0.7,

          'min_child_weight':0,

          'alpha':4,

          'objective': 'binary:logistic', 

          'scale_pos_weight':9,

          'eval_metric': 'auc', 

          'random_state': 99,

 #         'threads': 5,

          'silent': True}
#model = xgb.train(params, dtrain, 1000, watchlist, maximize=True, early_stopping_rounds=50, verbose_eval=10)

model = xgb.train(params, dtrain, 129, watchlist, maximize=True, verbose_eval=10)
dtest1 = xgb.DMatrix(test_df1[feature_cols])

dtest2 = xgb.DMatrix(test_df2[feature_cols])
sub = pd.read_csv('/content/kaggle/talkingdata/sample_submission.csv.zip', compression='zip')

sub.shape
sub['is_attributed'][:test_df1.shape[0]] = model.predict(dtest1, ntree_limit=model.best_ntree_limit)
sub['is_attributed'][test_df1.shape[0]:] = model.predict(dtest2, ntree_limit=model.best_ntree_limit)
sub.head()
sub.shape
sub.to_csv('sample_submission.csv',index=False)