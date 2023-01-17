# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json as js
from pandas.io.json import json_normalize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Any results you write to the current directory are saved as output.
# Reading train and test data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print("The shape of train data is :",train_data.shape)
print("The shape of test data is:",test_data.shape)
# Retrieving the json data from the actual data
json_cols = ['device','geoNetwork','totals','trafficSource']
json_data = train_data.loc[:,json_cols]
json_data.head()
# A basic function to get all the features enclosed in the json columns
def jcol_feats(col_name):
    j = js.loads(json_data[col_name][0])
    feats = j.keys()
    return feats
n_dev = len(jcol_feats('device'))
print("Number of features in device column:",n_dev)
n_geo= len(jcol_feats('geoNetwork'))
print("Number of features in geoNetwork column:",n_geo)
n_tot = len(jcol_feats('totals'))
print("Number of features in totals column:",n_tot)
n_traf = len(jcol_feats('trafficSource'))
print("Number of features in tafficSource column:",n_traf)
# https://www.kaggle.com/shivamb/exploratory-data-analysis-ga-customer-revenue

json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
def load_df(filename):
    path = "../input/" + filename
    df = pd.read_csv(path, converters={column: js.loads for column in json_cols}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
data = load_df("train.csv")
data.head()
len(data['totals_transactionRevenue'].value_counts()) # Number of users from whom google earned revenue
# Simple function for removing constant columns
def const(d):
    dt = d.loc[:,d.apply(pd.Series.nunique) != 1]
    return dt
act_data = const(data)
act_data.shape
def missing_data(data):  #calculates missing values in each column
    total = data.isnull().sum().reset_index()
    total.columns  = ['Feature_Name','Missing_value']
    total_val = total[total['Missing_value'] > 0]
    total_val = total.sort_values(by ='Missing_value',ascending= False)
    return total_val
missing_data(act_data).head(9) # 9 features in train data have missing value in them.
