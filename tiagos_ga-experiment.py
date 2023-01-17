# https://www.kaggle.com/codlife/pre-processing-for-huge-train-data-with-chunksize
# https://www.kaggle.com/usmanabbas/flatten-columns-remove-columns-before-normalizing
import os
import json
import numpy as np
from numpy import loadtxt
import pandas as pd
from pandas.io.json import json_normalize
from ast import literal_eval
import gc
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
gc.enable()
def load_df(csv_path):
    #csv contains nested data as JSON
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    dfs = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str','visitId': 'str'}, 
                    chunksize = 100000)
    chunk_count=0 #limit chunks
    #Ignored for now 
    #due to dimension ('customDimensions','hits')
    bad_columns = ['customDimensions','hits','date','trafficSource.campaign','socialEngagementType','trafficSource.adwordsClickInfo.gclId']
    #start loading chunks
    for df in dfs:
        gc.collect()
        df.reset_index(drop = True,inplace = True)
        #unnest JSON_COLUMNS
        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        #drop columns that contains irrelevant data like 'not available in demo dataset'
        droped_cols = []
        for col in df.columns:
            if 'not available in demo dataset'in df[col].unique() :
                    droped_cols = np.append(droped_cols,col)
        df = df.drop(axis='1', columns = droped_cols)
        #print(f'Droped {droped_cols}')
        #change type of columns for better handling
        df['totals.bounces']=df['totals.bounces'].apply(lambda x : x == '1')
        df['totals.newVisits']=df['totals.newVisits'].apply(lambda x : x == '1')
        df['trafficSource.isTrueDirect']=df['trafficSource.isTrueDirect'].apply(lambda x : x == True)
        df['trafficSource.adwordsClickInfo.isVideoAd']=df['trafficSource.adwordsClickInfo.isVideoAd'].apply(lambda x : x == True)
        #print(f"Droping columns {bad_columns}")
        df = df.drop(axis='1', columns = bad_columns)
        ans = pd.concat([ans, df], axis = 0,sort=False).reset_index(drop = True)
        print(f"Shape = Shape: {ans.shape}")
        chunk_count = chunk_count + 1 
        if(chunk_count == 10):
            break
    return ans
train = load_df('../input/train_v2.csv')
train.describe(include="object")
train.describe(include="bool")
pd.options.display.float_format = '{:.2f}'.format
train.describe(include="int64")
def date_preProcessing(df):
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['dayOfWeek'] = train['date'].dt.dayofweek
    df['quarter'] = train['date'].dt.quarter
    df['weekofyear'] = train['date'].dt.weekofyear
    print("Droping column ['date','visitStartTime']")
    df = df.drop(axis='1', columns = ['date','visitStartTime'])
    return df
train = date_preProcessing(train)
target = np.log1p(train['totals.transactionRevenue'].fillna(0).astype(float))
ids = train[['fullVisitorId']].copy()
features = train.drop(['fullVisitorId', 'visitId','totals.transactionRevenue'], axis=1, inplace=False)
for cols in features.columns:
    if features[cols].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(features[cols].values))
        features[cols] = lbl.transform(list(features[cols].values))
model = xgb.XGBRegressor()
parameters = {'nthread':[4], 
              'objective':['reg:linear'],
              'min_child_weight':[0.5,1,10],
               'max_depth':[3,5,10],
              'learning_rate': [.01, 0.05, .1], 
              'max_depth': [5, 6, 7],
             'n_estimators': [500,100,1000]}
xgb_grid = GridSearchCV(model,
                        parameters,
                        cv = 2,
                        n_jobs = 1,
                        verbose=True)
xgb_grid.fit(features,target)
print(f"Best R2 Score {xgb_grid.best_score_}")
print(f"Best Params {xgb_grid.best_params_}")