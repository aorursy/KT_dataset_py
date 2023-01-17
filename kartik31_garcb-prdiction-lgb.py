# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pandas.io.json import json_normalize
import json
import os
print(os.listdir("../input"))

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
# Any results you write to the current directory are saved as output.
train_df=load_df('../input/train.csv')
test_df=load_df('../input/test.csv')
print(train_df.head(),test_df.head())
train_df_copy=train_df
test_df_copy=test_df
train_df=train_df_copy
test_df=test_df_copy
drop_column=[]
for col in train_df.columns:
    if len(train_df[col].unique())==1:
        drop_column.append(col)
for col in test_df.columns:
    if len(test_df[col].unique())==1:
        if col not in drop_column:
            drop_column.append(col)
print(drop_column)
train_df=train_df.drop(drop_column,axis=1)
test_df=test_df.drop(drop_column,axis=1)
print(train_df.shape,test_df.shape)
#print(train_df.isna().any())
print([col for col in train_df.columns if col not in test_df.columns])
#'trafficSource.campaignCode' column of train data is not test data so  drop it 
train_df=train_df.drop('trafficSource.campaignCode',axis=1)
train_y=train_df['totals.transactionRevenue'].values
train_id=train_df['fullVisitorId'].values
test_id=test_df['fullVisitorId'].values
# Drop totals.transactionRevenue,fullVisitorId and sessionId(as it is unique for every session)
train_df=train_df.drop(['totals.transactionRevenue','sessionId'],axis=1)
test_df=test_df.drop(['sessionId'],axis=1)
for col in train_df:
    print(col,(train_df[col].isnull().sum()*100)/len(train_df[col]),train_df[col].dtype,len(test_df[col].unique()))
    if (train_df[col].isnull().sum()*100)/len(train_df[col])>65:
        train_df=train_df.drop(col,axis=1)
        test_df=test_df.drop(col,axis=1)
#train_df.isnull().sum(axis=0)
cat_data_columns=[]
for col in train_df.columns:
    if train_df[col].dtype=='object':
        cat_data_columns.append(col)
#print(cat_data_columns)
from sklearn.preprocessing import LabelEncoder
for col in cat_data_columns:
    lblen=LabelEncoder()
    lblen.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col]=lblen.transform(train_df[col].values.astype('str'))
    test_df[col]=lblen.transform(test_df[col].values.astype('str'))
train_df_copy["totals.transactionRevenue"].fillna(0, inplace=True)
print(train_df_copy["totals.transactionRevenue"].isna().any())
train_y=train_df_copy['totals.transactionRevenue'].values
print(train_y)
#train_df=train_df.drop('totals.transactionRevenue',axis=1)
train_df.insert(0,"totals.transactionRevenue",train_y)
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
#print(train_df['date'])
dev_df = train_df
val_df = train_df[train_df['date']>20170531]
train_y = np.log1p(dev_df["totals.transactionRevenue"].values.astype(float))
val_y = np.log1p(val_df["totals.transactionRevenue"].values.astype(float))

dev_df_x=dev_df.drop(['totals.transactionRevenue','fullVisitorId'],axis=1)
val_df_x=val_df.drop(['totals.transactionRevenue','fullVisitorId'],axis=1)
train_x=dev_df_x
val_x=val_df_x
test_x=test_df
train_x.info()
val_x.info()
print(len(train_y))
len(val_y)
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
train_set=lgb.Dataset(train_x, label=train_y)
val_set=lgb.Dataset(val_x,label=val_y)
params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
lgb_model=lgb.train(params,train_set,500,valid_sets=[val_set],early_stopping_rounds=100, verbose_eval=100)
pred_test = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
pred_val = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
#from sklearn import metrics
pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)

#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue","PredictedRevenue"].sum().reset_index()
#val_pred_df["PredictedRevenue"] = val_pred_df.groupby("fullVisitorId")["PredictedRevenue"].sum().reset_index()
print(val_pred_df.head())
val_pred_col=np.log1p(val_pred_df["transactionRevenue"].values)
val_actual_col=np.log1p(val_pred_df["PredictedRevenue"].values)
print(np.sqrt(metrics.mean_squared_error(val_pred_col,val_actual_col)))
sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("result_lgb_full.csv", index=False)
sub_df.head()