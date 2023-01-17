import pandas as pd
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import lightgbm as lgb
import matplotlib.pyplot as plt
import json
import pickle
import sys
import math
import gc

from pandas.io.json import json_normalize
from datetime import datetime

import os
print(os.listdir("../input"))
def load(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows,)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
train = load('../input/train.csv')
test = load('../input/test.csv')
def to_numeric(data):
    for feature in data.dtypes[data.dtypes == 'object'].index: 
        data[feature] = pd.to_numeric(data[feature],errors='ignore')
to_numeric(train)
to_numeric(test)
# Customers present in train and test
repeat_clients = set(train.fullVisitorId) & set(test.fullVisitorId)
len(set(train.fullVisitorId)), len(repeat_clients)
train['isBuy'] = train['totals.transactionRevenue'] > 0 
train['isBuy'].mean(), train[train.fullVisitorId.isin(repeat_clients)].isBuy.mean()
# Drop columns with constant value 
const_columns = [column for column in train.columns if len(train[column].value_counts(dropna=False)) == 1]
train.drop(columns=const_columns, axis=1, inplace=True)
test.drop(columns=const_columns, axis=1, inplace=True)
# Drop columns containing only in train
set(train.columns) ^ set(test.columns)
del train['trafficSource.campaignCode']
def get_time_feature(df):
    df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df.visitStartTime = pd.to_datetime(df.visitStartTime, unit='s')
    df['hour'] = df.visitStartTime.dt.hour
    df['dayofweek'] = df.visitStartTime.dt.dayofweek
    df['weekofyear'] = df.visitStartTime.dt.weekofyear
    df['month'] = df.visitStartTime.dt.month
get_time_feature(train)
get_time_feature(test)
class ThrColumnEncoder:
    """
    The threshold label encoder. 
    To avoid overfitting we can combine rare values into one group. Class work with pd.Series.
    thr: Threshold as a percentage, values whose number is less than the threshold will be replaced by a single label.
    """
    def __init__(self, thr=0.5):
        self.thr = thr
        self.categories = defaultdict(lambda:-1) # Those values that are X_test, but not in X_train will be replaced by -1.
        
    def fit(self, x):
        values = x.value_counts(dropna=False)*100/len(x)
        for value, key in enumerate(values[values >= self.thr].index):
            self.categories[key] = value
        for value, key in enumerate(values[values < self.thr].index):
            self.categories[key] = -1 # Rare values replace -1
            
    def transform(self, x):   
        return x.apply(self.categories.get)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
class ThrLabelEncoder:
    """
    Work with pd.DataFrame.
    """
    def __init__(self, thr=0.5):
        self.thr = thr
        self.column_encoders = {}
        self.features = None
        
    def fit(self, X):
        self.features = X.columns
        for feature in self.features:
            ce = ThrColumnEncoder()
            ce.fit(X[feature])
            self.column_encoders[feature] = ce
            
    def transform(self, X):
        X = X.copy()
        for feature in self.features: 
            ce = self.column_encoders[feature]
            X.loc[:, feature] = ce.transform(X[feature])
        return X
            
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
id_cols = 'fullVisitorId', 'sessionId'

tfidf_cols = 'geoNetwork.networkDomain', 'trafficSource.referralPath', 'trafficSource.source'

cat_cols = list(set(train.dtypes[train.dtypes == 'object'].index) - set(id_cols) - set(tfidf_cols)  |  set(('dayofweek', 'hour', 'month')))
class DummyEncoder(ThrLabelEncoder):
    """
    For each unique value of a categorical feature, we can create our own binary feature.
    Generally speaking, this transformation is not necessary for gradient boosting.
    But for this task, I decided to move from the feature description of the visit to the feature description
    of the client and without the dummy-encoding can not do it.
    """
    def transform(self, X):
        result = []
        for feature in self.features: 
            ce = self.column_encoders[feature]
            tf_feature = ce.transform(X[feature])
            popular_values = [value for key, value in self.column_encoders[feature].categories.items() if value != -1]
            popular_keys = [key for key, value in self.column_encoders[feature].categories.items() if value != -1]
            columns = ['%s_%s'%(feature, key) for key in popular_keys + ['rare']]
            feature_dummies = pd.concat([tf_feature == value for value in popular_values] + [tf_feature == -1], axis=1)
            feature_dummies.columns = columns
            result.append(feature_dummies)
        return pd.concat(result, axis=1)
de = DummyEncoder()
train_de = de.fit_transform(train[cat_cols])
test_de = de.transform(test[cat_cols])

train_de.index = train.fullVisitorId
test_de.index = test.fullVisitorId
def life_time(x):
    return (x.max() - x.min()).total_seconds()
aggregates = {'totals.pageviews': [sum, min, max, np.mean], 
              'totals.hits': [sum, min, max, np.mean], 
              'date': life_time,
              'visitNumber': [max, min],
              'totals.bounces': sum,
              'totals.transactionRevenue': sum
             }
%time train_gr = train.groupby('fullVisitorId').agg(aggregates)
def groupby_rename(df):
    df.columns = ['%s_%s'% (df.columns.levels[0][i],df.columns.levels[1][j]) for i,j in \
                  zip(df.columns.labels[0], df.columns.labels[1])]
groupby_rename(train_gr)
del aggregates['totals.transactionRevenue']
%time test_gr = test.groupby('fullVisitorId').agg(aggregates)
groupby_rename(test_gr)
train_de_sum = train_de.groupby(level=0).sum()

test_de_sum = test_de.groupby(level=0).sum()

X = pd.concat([train_de_sum, train_gr], axis=1)
X_test = pd.concat([test_de_sum, test_gr], axis=1)

Y = np.log1p(X['totals.transactionRevenue_sum'])
del X['totals.transactionRevenue_sum']
del train_de_sum
del test_de_sum
del train_gr
del test_gr
del train_de
del test_de
class TFIDFER:
    """
    Class encapsulating tfidf transformation and renaming column of pd.Dataframe with using name tfidf-features.
    """
    def __init__(self, max_df=0.9, min_df=0.01, max_features=100, ngram_range=(1,2)):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features  = max_features
        self.ngram_range = ngram_range
        
        self.column_encoders = {}
        self.features = None
        
    def fit(self, X): 
        self.features = X.columns
        for feature in self.features:
            tfidf = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.max_features, ngram_range=self.ngram_range)
            tfidf.fit(X[feature])
            self.column_encoders[feature] = tfidf
            
    def transform(self, X):
        result = []
        for feature in self.features: 
            items_tfidf = pd.DataFrame(self.column_encoders[feature].transform(X[feature]).toarray(), X.index)
            col_names = [word.replace(' ','_') for word, index in sorted(self.column_encoders[feature].vocabulary_.items(), key = lambda x:x[1])]
            items_tfidf.columns =  ['tfidf_%s_%s'%(feature, name) for name in col_names] 
            result.append(items_tfidf.copy())
        return pd.concat(result, axis=1)
            
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
def get_tfidf(df):
    df['geoNetwork.networkDomain'] = df['geoNetwork.networkDomain'].apply(lambda x: x.replace('.', ' ').replace(':', ' ') + ' ')
    df['trafficSource.source'] = df['trafficSource.source'].apply(lambda x: x.replace('.', ' ').replace(':', ' ') + ' ')
    df['trafficSource.referralPath'] = df['trafficSource.referralPath'].astype(str).apply(lambda x: x.replace('/', ' ') + ' ')
    aggregates = {'geoNetwork.networkDomain': sum, 
                  'trafficSource.source': sum, 
                  'trafficSource.referralPath': sum}
    return df.groupby('fullVisitorId').agg(aggregates)
aggregates = {'geoNetwork.networkDomain': sum, 
              'trafficSource.source': sum, 
              'trafficSource.referralPath': sum}
%time train_ftidf_sum = get_tfidf(train)
%time test_ftidf_sum = get_tfidf(test)
tfidfer = TFIDFER()
train_tfidf = tfidfer.fit_transform(train_ftidf_sum)
test_tfidf = tfidfer.transform(test_ftidf_sum)
X_test = pd.concat([X_test, test_tfidf], axis=1)
X = pd.concat([X, train_tfidf],axis =1)
del train 
del test 
gc.collect()
X_test.head()
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.33, random_state=0)
gbm = lgb.LGBMRegressor(objective = 'regression',  
                        max_depth = 11,
                        colsample_bytre = 0.8,
                        subsample = 0.8, 
                        learning_rate = 0.1,
                        n_estimators = 300)
gbm.fit(X_train, Y_train, 
        eval_set=[(X_valid, Y_valid)],
        eval_metric='rmse',
        early_stopping_rounds=5)
%matplotlib inline
lgb.plot_importance(gbm, max_num_features=120, figsize=(10,40))
Y_test = pd.Series(gbm.predict(X_test),index= X_test.index)

Y_test[Y_test<0] = 0

Y_test.name = "PredictedLogRevenue"

Y_test.head()