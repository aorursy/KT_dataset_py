import sys
import warnings
import numpy as np
import pandas as pd 
import json
import time
import ast
from sklearn import metrics
warnings.filterwarnings('ignore')

train_v2_file = '../input/train_v2.csv'
!ls -ld $train_v2_file
relevant_cols = ['date', 'fullVisitorId', 'totals']
train = pd.read_csv(train_v2_file, usecols=relevant_cols, dtype={'fullVisitorId': 'str'})
train['totals.transactionRevenue'] = train['totals'].apply(lambda x: ast.literal_eval(x).get('transactionRevenue',np.nan))
train['totals.transactionRevenue'] = pd.to_numeric(train['totals.transactionRevenue'], errors="coerce")
train.drop(columns=['totals'], inplace=True)
print (train.shape)
print (train.columns)
def rmse_log1p(df, col1, col2):
    return np.sqrt(metrics.mean_squared_error(np.log1p(df[col1].values), np.log1p(df[col2].values)))

print ("")
print ("Get 5 month period for test submission and 2 month period for competition:")
keepcols = ['fullVisitorId','totals.transactionRevenue']
submission2017  = train.loc[(20170501 <= train['date'])].loc[(train['date'] <= 20171005)][keepcols]
competition2017 = train.loc[(20171201 <= train['date'])].loc[(train['date'] <= 20180131)][keepcols]
print ("submission2017", submission2017.shape)
print ("competition2017", competition2017.shape)

print ("")
print ("Get visitors who are in both submission and competition periods:")
submission_visitors = list(submission2017['fullVisitorId'].dropna().astype(str).unique())
competition_visitors = list(competition2017['fullVisitorId'].dropna().astype(str).unique())
visitors_in_both = set(submission_visitors) & set(competition_visitors)
print ("unique 2017 sub visitors:",len(submission_visitors))
print ("unique 2017 comp visitors:",len(competition_visitors))
print ("unique 2017 in both:",len(visitors_in_both))

print ("")
submission2017['totals.transactionRevenue'] = 0.0
submission2017['totals.predictedRevenue'] = 0.0
submission2017 = submission2017.groupby('fullVisitorId').sum().reset_index()
print ("submission with unique visitors and predictions of zero:", submission2017.shape)

competition2017['totals.transactionRevenue'].fillna(0, inplace=True)
competition2017 = competition2017.groupby('fullVisitorId').sum().reset_index()
competition2017 = competition2017[competition2017['fullVisitorId'].isin(visitors_in_both)]
competition2017['totals.predictedRevenue'] = 0.0
print ("competition visitors who appeared in submission:", competition2017.shape)

submission2017 = pd.concat([submission2017, competition2017], axis=0)
submission2017 = submission2017.groupby('fullVisitorId').sum().reset_index()
print ("submission with competiton data for calculating RMSE:", submission2017.shape)

print ("")
print("RMSE score with all predictions zero:", rmse_log1p(submission2017, 'totals.transactionRevenue', 'totals.predictedRevenue'))