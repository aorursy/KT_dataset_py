import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import datetime
import os
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
%matplotlib inline
print(os.listdir("../input"))
json_columns = ['totals']
usecols = [ 'date', 'fullVisitorId', 'totals']
fillnazero_columns = ['totals_totalTransactionRevenue', 'totals_transactionRevenue', 'totals_transactions']
float_columns = ['totals_totalTransactionRevenue', 'totals_transactionRevenue', 'totals_transactions']
train = pd.read_csv('../input/train_v2.csv', parse_dates=['date'], converters={column: json.loads for column in json_columns}, 
                    dtype={'fullVisitorId': 'str'}, usecols=usecols)
def process_json_columns(df, json_columns):
    for column in json_columns:
        normalized_column_df = json_normalize(df[column]) 
        normalized_column_df.columns = [f"{column}_{subcolumn}" for subcolumn in normalized_column_df.columns] 
        df = df.drop(column, axis=1).merge(normalized_column_df, right_index=True, left_index=True)
    return df

def fillna(df, fillnazerocols):
    df[fillnazerocols] = df[fillnazerocols].fillna(0)
    return df

def change_dtypes_float(df, float_columns):
    for c in float_columns:
        df[c] = df[c].astype('float32')
    return df

train = process_json_columns(train, json_columns)
train = change_dtypes_float(train, float_columns)
train = fillna(train, fillnazero_columns)
private = train[(train['date'] >= datetime.datetime(2017, 12, 1)) & (train['date'] <= datetime.datetime(2018, 1, 31))] 
test = train[(train['date'] >= datetime.datetime(2017, 5, 1)) & (train['date'] <= datetime.datetime(2017, 10, 15))] 
gap = train[(train['date'] >= datetime.datetime(2017, 10, 16)) & (train['date'] <= datetime.datetime(2017, 11, 30))] 
def analysis_plot(private, test, gap):
    fullVisitorId_private = pd.Index(private['fullVisitorId'].unique())
    fullVisitorId_test = pd.Index(test['fullVisitorId'].unique())
    private_revenue = private.groupby('fullVisitorId')['totals_totalTransactionRevenue'].sum()
    gap_revenue = gap.groupby('fullVisitorId')['totals_totalTransactionRevenue'].sum()

    data = [go.Bar(
        x=[ '#Visitors in test that are in private', '#Visitors in private with revenue', 
            '#Visitors in private with revenue that are in test', '#Visitors in private with revenue that are in gap'],
        y=[fullVisitorId_test.isin(fullVisitorId_private).sum(),
            (private_revenue>0).sum(),
            private_revenue[private_revenue>0].index.isin(test['fullVisitorId']).sum(),
            gap_revenue[gap_revenue>0].index.isin(test['fullVisitorId']).sum()]                                                                                
    )]
    py.iplot(data, filename='basic-bar')
analysis_plot(private, test, gap)
test = pd.read_csv('../input/test_v2.csv', parse_dates=['date'], converters={column: json.loads for column in json_columns}, 
                    dtype={'fullVisitorId': 'str'}, usecols=usecols)
test = process_json_columns(test, json_columns)
test = change_dtypes_float(test, float_columns)
test = fillna(test, fillnazero_columns)
test['date'].min()
test['date'].max()
full = train.append(test)
private_recent = full[(full['date'] >= datetime.datetime(2018, 8, 15)) & (full['date'] <= datetime.datetime(2018, 10, 15))] 
test_recent = full[(full['date'] >= datetime.datetime(2018, 1, 1)) & (full['date'] <= datetime.datetime(2018, 5, 13))] 
gap_recent = full[(full['date'] >= datetime.datetime(2018, 6, 14)) & (full['date'] <= datetime.datetime(2018, 8, 14))] 
analysis_plot(private_recent, test_recent, gap_recent)