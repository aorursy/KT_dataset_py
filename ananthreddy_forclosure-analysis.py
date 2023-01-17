# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sys

import re

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from tqdm import tqdm

warnings.filterwarnings('ignore')
rf_data = pd.read_excel('../input/RF_Final_Data.xlsx')

print (rf_data.shape)

rf_data.head()
rf_data.isnull().sum()
rf_data = rf_data.drop(['Preprocessed_EmailBody','Preprocessed_Subject'], 1)
cust_data = pd.read_excel('../input/Customers_31JAN2019.xlsx')

print (cust_data.shape)

cust_data.head()
cust_data.isnull().sum()
cust_data = cust_data.drop(['PROFESSION','OCCUPATION','POSITION','PRE_JOBYEARS'], 1)
lms_data = pd.read_excel('../input/LMS_31JAN2019.xlsx')

print (lms_data.shape)

lms_data.head()
lms_data['CITY'].value_counts()
lms_data['CITY'].nunique()
lms_data['PRODUCT'].value_counts()
lms_data.isnull().sum()
lms_data = lms_data.drop(['NPA_IN_LAST_MONTH', 'NPA_IN_CURRENT_MONTH', 'CITY'], 1)
lms_data.AGREEMENTID.nunique()
cat_columns = ['PRODUCT']

lms_data = pd.get_dummies(columns = cat_columns, data = lms_data)
useless_columns = ['INTEREST_START_DATE', 'AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE', 'SCHEMEID']

lms_data = lms_data.drop(useless_columns, 1)
lms_data.head()
lms_data = lms_data.groupby(['AGREEMENTID']).mean()

lms_data = lms_data.reset_index()
lms_data.head()
train_data = pd.read_csv('../input/train_foreclosure.csv')

print (train_data.shape)

train_data.head()
test_data = pd.read_csv('../input/test_foreclosure.csv')

print (test_data.shape)

test_data.head()
lms_data.head()
lms_train_data = pd.merge(train_data, lms_data, on = 'AGREEMENTID')

lms_train_data.shape
lms_train_data.head()
lms_train_data.isnull().sum()
mean_value = lms_train_data['LAST_RECEIPT_AMOUNT'].mean()

lms_train_data['LAST_RECEIPT_AMOUNT'] = lms_train_data['LAST_RECEIPT_AMOUNT'].fillna(value = mean_value)
lms_train_data = lms_train_data.drop(['CUSTOMERID'], 1)
lms_test_data = pd.merge(test_data, lms_data, on = 'AGREEMENTID')

lms_test_data.shape
lms_test_data.head()
lms_test_data.isnull().sum()
lms_test_data = lms_test_data.drop(['CUSTOMERID'], 1)

mean_value = lms_test_data['LAST_RECEIPT_AMOUNT'].mean()

lms_test_data['LAST_RECEIPT_AMOUNT'] = lms_test_data['LAST_RECEIPT_AMOUNT'].fillna(value = mean_value)
y = lms_train_data['FORECLOSURE']

X_ID = lms_train_data['AGREEMENTID']

X = lms_train_data.drop(['FORECLOSURE', 'AGREEMENTID'], 1)
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb
xgb.fit(X, y)
X_test_ID = lms_test_data['AGREEMENTID']

X_test = lms_test_data.drop(['AGREEMENTID', 'FORECLOSURE'], 1)
xgb_pred = xgb.predict(X_test)
submission = pd.DataFrame(columns = ['AGREEMENTID', 'FORECLOSURE'])

submission['AGREEMENTID'] = X_test_ID

submission['FORECLOSURE'] = xgb_pred
submission.to_csv('submission.csv', index = False)