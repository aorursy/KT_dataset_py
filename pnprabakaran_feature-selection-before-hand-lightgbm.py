# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, json
from pandas.io.json import json_normalize
import lightgbm as lgb
from sklearn.feature_selection import RFE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# configurations
FILE_DIR = "../input"
file_name = "train.csv"
nrows = None
JSON_COLS_TO_PARSE = ['device', 'geoNetwork', 'totals', 'trafficSource']
file_path = os.path.join(FILE_DIR, file_name)
# data load
df = pd.read_csv(file_path,
    converters={column: json.loads for column in JSON_COLS_TO_PARSE}, 
    dtype={'fullVisitorId': 'str'},  # Important!!
    nrows=nrows) 
df.info()
#json parsing
for cols in JSON_COLS_TO_PARSE:
    column_as_df = json_normalize(df[cols])
    column_as_df.columns = [f"{cols}_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop(cols, axis=1).merge(column_as_df, right_index=True, left_index=True)

df.info()
# select columns to transform

# numeric cols
numeric_cols_to_transform = ['totals_hits'
                ,'totals_pageviews'
                ,'visitNumber'
                ,'totals_newVisits'
                ,'totals_bounces'
                ,'totals_transactionRevenue']
#object cols
obj_cols_to_transform = list(df.select_dtypes(include=['object', 'bool']).columns)

# transfer numeric columns as float
df[numeric_cols_to_transform] = df[numeric_cols_to_transform].fillna(0)
for col in numeric_cols_to_transform:
    df[col] = df[col].astype('float')
                        
# transfer object columns as category cols
for col in obj_cols_to_transform:
    df[col] =  df[col].astype('category').cat.codes
df.info()
# split train and target variables
train = df.drop('totals_transactionRevenue', axis=1)
target = df.totals_transactionRevenue
# Feature importance

#lightGBM model fit
gbm = lgb.LGBMRegressor()
gbm.fit(train, target)
gbm.booster_.feature_importance()

# importance of each attribute
fea_imp_ = pd.DataFrame({'cols':train.columns, 'fea_imp':gbm.feature_importances_})
fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)

#Recursive Feature Elimination(RFE)

# create the RFE model and select 10 attributes
rfe = RFE(gbm, 10)
rfe = rfe.fit(train, target)

# summarize the selection of the attributes
print(rfe.support_)

# summarize the ranking of the attributes
fea_rank_ = pd.DataFrame({'cols':train.columns, 'fea_rank':rfe.ranking_})
fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending = True)
