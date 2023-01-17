import os
import numpy as np 
import pandas as pd 
import json
from pandas.io.json import json_normalize
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
import gc
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
%matplotlib inline
PATH="../input/"
 
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']

def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}.{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)

    return data_df
    
def process_date_time(data_df):
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month.unique.user.count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day.unique.user.count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday.unique.user.count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    return data_df

def process_format(data_df):
    for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource.isTrueDirect'].fillna(False, inplace=True)
    return data_df
    
def process_device(data_df):
    data_df['browser.category'] = data_df['device.browser'] + '.' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '.' + data_df['device.operatingSystem']
    return data_df

def process_totals(data_df):
    data_df['visitNumber'] = (data_df['visitNumber'])
    data_df['totals.hits'] = (data_df['totals.hits'])
    data_df['totals.pageviews'] = (data_df['totals.pageviews'].fillna(0))
    data_df['mean.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('mean')
    data_df['sum.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('sum')
    data_df['max.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('max')
    data_df['min.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('min')
    data_df['var.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('var')
    data_df['mean.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('mean')
    data_df['sum.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('sum')
    data_df['max.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('max')
    data_df['min.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('min')    
    return data_df

def process_geo_network(data_df):
    data_df['sum.pageviews.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    data_df['count.pageviews.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    data_df['mean.pageviews.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    data_df['sum.hits.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    data_df['count.hits.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    data_df['mean.hits.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')
    return data_df

def process_traffic_source(data_df):
    data_df['source.country'] = data_df['trafficSource.source'] + '.' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '.' + data_df['trafficSource.medium']
    data_df['medium.hits.mean'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('mean')
    data_df['medium.hits.max'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('max')
    data_df['medium.hits.min'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('min')
    data_df['medium.hits.sum'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('sum')
    return data_df
train_df = read_parse_dataframe('train.csv')
test_df = read_parse_dataframe('test.csv')
train_df.columns
cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
train_df.drop(['trafficSource.campaignCode'], axis=1, inplace=True)
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype(float)
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0)
train_df['totals.transactionRevenue'] = np.log1p(train_df['totals.transactionRevenue'])
train_df = process_date_time(train_df)
train_df = process_format(train_df)
train_df = process_device(train_df)
train_df = process_totals(train_df)
train_df = process_geo_network(train_df)
train_df = process_traffic_source(train_df)

test_df = process_date_time(test_df)
test_df = process_format(test_df)
test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)
test_df = process_traffic_source(test_df)
num_cols = ['month.unique.user.count', 'day.unique.user.count', 'weekday.unique.user.count',
            'visitNumber', 'totals.hits', 'totals.pageviews', 
            'mean.hits.per.day', 'sum.hits.per.day', 'min.hits.per.day', 'max.hits.per.day', 'var.hits.per.day',
            'mean.pageviews.per.day', 'sum.pageviews.per.day', 'min.pageviews.per.day', 'max.pageviews.per.day',
            'sum.pageviews.per.network.domain', 'count.pageviews.per.network.domain', 'mean.pageviews.per.network.domain',
            'sum.hits.per.network.domain', 'count.hits.per.network.domain', 'mean.hits.per.network.domain',
            'medium.hits.mean','medium.hits.min','medium.hits.max','medium.hits.sum']
                
not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
                 "visitId", "visitStartTime", 'totals.transactionRevenue', 'trafficSource.referralPath']
cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
for col in num_cols:
    train_df[col] = np.log1p((train_df[col].values))
    test_df[col] = np.log1p((test_df[col].values))
x = pd.concat([train_df,test_df],sort=False)
x = x.reset_index(drop=True)
for col in num_cols:
    x.loc[:,col] = pd.cut(x[col], 50,labels=False)
test_df = x.loc[train_df.shape[0]:].copy().reset_index(drop=True)
train_df = x.loc[:train_df.shape[0]].copy().reset_index(drop=True)
for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
train_df.fillna(0,inplace=True,axis=1)
test_df.fillna(0,inplace=True,axis=1)
test_df['totals.transactionRevenue'] = 0.
features = num_cols+cat_cols
categories = features[:]
numerics = []
currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train_df.shape[0]
noofcolumns = len(features)
with open("alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = train_df.iloc[r].to_dict()
        datastring += str(float(datarow['totals.transactionRevenue']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)
        
noofrows = test_df.shape[0]
noofcolumns = len(features)
with open("alltestffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = test_df.iloc[r].to_dict()
        datastring += str(float(datarow['totals.transactionRevenue']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)