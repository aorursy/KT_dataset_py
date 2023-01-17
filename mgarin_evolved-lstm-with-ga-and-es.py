# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
import matplotlib.pyplot as plt
import matplotlib
import re
from scipy import stats

matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['font.size'] = 12

import random
random.seed(1)
import time

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(mtrain, ntrain) = env.get_training_data()
market = mtrain.copy()
news = ntrain.copy()
market.time = mtrain.time.astype('datetime64[D, UTC]')

news.time = mtrain.time.astype('datetime64[D, UTC]')

print(market.shape)
print(news.shape)
market.head()
print("Total asset count: " + str(len(market['assetCode'].unique())))

def get_asset(df, asset=None):
    #get an asset, if none specified get random asset
    ass = asset
    if ass is None: #get random asset
        ass = df['assetCode'].unique()[random.randint(0, len(df['assetCode'].unique()))]
    ass_market = df[df['assetCode'] == ass]
    ass_market.index = ass_market.time
    return ass_market
plt.plot(get_asset(market).close) #Plots asset
#gets a sample of assets with all data present after 2009
#since a lot of companies went bankrupt then lol
def market_split(market, sample_size=100000):
    midx = market[market.time > '2009'][['time', 'assetCode']]
    midx = midx.sample(sample_size)
    midx = midx.sort_values(by=['time'])
    
    market_train, market_test = train_test_split(midx, shuffle=False, random_state=24)
    market_train, market_val = train_test_split(market_train, test_size=0.1, shuffle=False, random_state=24)
    
    return market_train, market_val, market_test
mtrain, mval, mtest = market_split(market)
print("market: ")
print("    train size: " + str(len(mtrain)))
print("    val size:   " + str(len(mval)))
print("    test size:  " + str(len(mtest)))

print(str(len(mtrain.assetCode.unique())))
class MarketPrepro:
    
    assetcode_encoded = []
    assetcode_train_count = 0
    time_cols = ['year', 'week', 'day', 'dayofweek']
    num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']
    #all features
    feat_cols = ['assetCode_encoded'] + time_cols + num_cols
    
    label_cols = ['returnsOpenNextMktres10']
    
    def __init__(self):
        self.cats = {}
    
    def fit(self, mtrain):
        df = mtrain.copy()
        #fix/clean data
        mtrain = self.fix_train(mtrain)
        
        #get time cols
        mtrain = self.prep_time_cols(mtrain)
        
        #standardize features by using z = (x - u) / s
        self.num_scaler = StandardScaler()
        self.num_scaler.fit(mtrain[self.num_cols + self.time_cols].astype(float))
    
        mtrain = self.encode_asset(mtrain, True)
    
    def fix_train(self, mtrain):
        #fix/clean data
        max_ratio = 2 #removing outliers
        mtrain = mtrain[((mtrain['close'])/mtrain['open']).abs() <= max_ratio].loc[:]
        
        mtrain = self.safe_fix(mtrain)
        return mtrain
    
    def safe_fix(self, mtrain):
        #fill na and outliers, safe for train, no rows removed
        
        #fill na using bfill 
        mtrain[self.num_cols] = mtrain[['assetCode'] + self.num_cols].groupby('assetCode').transform(lambda g: g.fillna(method='bfill'))
        mtrain[self.num_cols] = mtrain[self.num_cols].fillna(0) #using 0
        
        #fix outliers based on quantiles
        mtrain[self.num_cols] = mtrain[self.num_cols].clip(mtrain[self.num_cols].quantile(0.01), mtrain[self.num_cols].quantile(0.99), axis=1)
        
        return mtrain
    
    def get_X(self, mtrain):
        #return x 
        mtrain = mtrain.copy()
        mtrain = self.safe_fix(mtrain)
        
        mtrain = self.prep_time_cols(mtrain)
        mtrain = self.encode_asset(mtrain, istrain=False)
        
        mtrain[self.num_cols + self.time_cols] = self.num_scaler.transform(mtrain[self.num_cols +self.time_cols].astype(float))
        
        return mtrain[self.feat_cols]
    
    def get_y(self, mtrain):
        y = (mtrain[self.label_cols]>=0).astype(float)
        return y
    
    def encode_asset(self, df, istrain):
        def encode(assetcode):
            try: 
                indx_val = self.assetcode_encoded.index(assetcode) + 1
            except ValueError: 
                self.assetcode_encoded.append(assetcode)
                indx_val = len(self.assetcode_encoded)
            
            indx_val = indx_val/ (self.assetcode_train_count + 1)
            return indx_val
        
        if istrain:
            self.assetcode_train_count = len(df['assetCode'].unique()) +1 
        df['assetCode_encoded'] = df['assetCode'].apply(lambda assetcode: encode(assetcode))
        return df
    
    def prep_time_cols(self, df): 
        #extract time cols, important for time series
        df = df.copy()
        df['year'] = df['time'].dt.year
        df['day'] = df['time'].dt.day
        df['week'] = df['time'].dt.week
        df['dayofweek'] = df['time'].dt.dayofweek
        return df
    
market_prepro = MarketPrepro()
print('market preprocessed lmao')
        
    
        
    
    
    
