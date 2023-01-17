import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics as ss

import json
import pandas.io.json as pdjson
import ast # Abstract Syntax Trees : The ast module helps Python applications to process trees of the Python abstract syntax grammar.
import datetime as dt

import gc   # Garbage Collector : gc exposes the underlying memory management mechanism of Python
gc.enable()

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import sklearn.metrics as sklm

import os
print(os.listdir("../input"))

def extraction(df): # here we declare a function to do all feature extraction as per Part 2.
    
    df['date'] = df['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    #% feature representation
    df.date = pd.to_datetime(df.date, errors='coerce')
    #% feature extraction - time and date features
    # Get the month value from date
    df['month'] = df['date'].dt.month
    # Get the week value from date
    df['week'] = df['date'].dt.week
    # Get the weekday value from date
    #df['weekday'] = df['date'].dt.weekday
    # Get the year
    df['year'] = df['date'].dt.year
    # Get the day of the month
    df['day'] = df['date'].dt.day
    # period counter
    df['count'] = ((df['year'] - 2016) * 24) + (df['month']-1)*2 + round( df['day']/30 ) - 13
    # drop date
    df = df.drop(labels=['date'], axis=1)
    
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['hour'] = df['visitStartTime'].dt.hour
    df = df.drop(labels=['visitStartTime'], axis=1)
    
    return df

def make_countsum(df, dfstr):
    df[dfstr] = df[dfstr].astype('str')
    
    df['totals_hits']=df['totals_hits'].fillna(0).astype('int')
    df['totals_pageviews']=df['totals_pageviews'].fillna(0).astype('int')
    
    df[str(dfstr+'_count')] = df[dfstr]
    df[str(dfstr+'_count')]=df.groupby(dfstr).transform('count')
    
    df[str(dfstr+'_hitssum')] = df.groupby(dfstr)['totals_hits'].transform('sum')
    df[str(dfstr+'_viewssum')] = df.groupby(dfstr)['totals_pageviews'].transform('sum')
    del(df[dfstr])
    return df


def cardinality_redux(df): # this function covnerts high cardinality categorical features to numeric aggregates
    lst = ['geoNetwork_city', 'geoNetwork_metro', 'geoNetwork_region', 'geoNetwork_country', 
           'geoNetwork_networkDomain', 'hits_appInfo.exitScreenName', 
           'hits_appInfo.landingScreenName', 'hits_appInfo.screenName', #'hits_eventInfo.eventLabel', 
           'hits_page.pagePath', 'hits_page.pagePathLevel1', 'hits_page.pagePathLevel2', 
           'hits_page.pagePathLevel3', 'hits_page.pagePathLevel4', 'hits_page.pageTitle', 
           'hits_referer', 'trafficSource_adContent', 'trafficSource_adwordsClickInfo.gclId']
    for dfstr in lst:
        df = make_countsum(df, dfstr)
    return df


def aggregate(df, col, leave): # fn to aggregate all categories in df[col] except for cols in leave
    df[col] = df[col].astype('str')
    include = df[col].unique()  # array of all unique categories
    include = list(include)
    include = set(include).difference(set(leave))  # set: take out 'leave' from include
    include = list(include)
    df.loc[df[col].isin(include), col] = "grouped"  # rename all cols in 'include' to 'grouped'
    return df

def all_agg(data):
    data = aggregate(data, 'device_operatingSystem', leave=['Windows', 'Macintosh', 'Android', 'iOS', 'Linux', 'Chrome OS'])
    data = aggregate(data, 'trafficSource_referralPath', leave=['/'])
    return data

def encode_num(data):
    data['hits_hitNumber'] = pd.to_numeric(data['hits_hitNumber'], errors='coerce', downcast='unsigned')
    data['hits_hour'] = pd.to_numeric(data['hits_hour'], errors='coerce', downcast='unsigned')
    data['totals_timeOnSite'] = pd.to_numeric(data['totals_timeOnSite'], errors='coerce', downcast='unsigned')
    return data

def clean(data):
    # Convert to object:
    data['hits_isEntrance'] = data['hits_isEntrance'].astype('str')
    data['hits_isExit'] = data['hits_isExit'].astype('str')
    data['hits_promotionActionInfo.promoIsView'] = data['hits_promotionActionInfo.promoIsView'].astype('str')
    data['trafficSource_adwordsClickInfo.isVideoAd'] = data['trafficSource_adwordsClickInfo.isVideoAd'].astype('str')
    data['trafficSource_isTrueDirect'] = data['trafficSource_isTrueDirect'].astype('str')
    
    # Encode nans as 0:
    data['totals_sessionQualityDim'].fillna(0, inplace=True)
    data['totals_timeOnSite'].fillna(0, inplace=True)
    
    # replace nans with mode:
    data['hits_hitNumber'].fillna(ss.mode(data['hits_hitNumber']), inplace=True)
    data['hits_hour'].fillna(ss.mode(data['hits_hour']), inplace=True)
    return data


def onehot(data):                    # TODO this is not OHE yet
    for col in data.columns:
        x=data[col]
        if (x.apply(np.isreal).all(axis=0)) & ((str(x.dtypes) != 'category')): # if numeric, but not category
            #print(col, 'is numeric')
            1+1
        else:
            data[col] = data[col].astype('category',copy=False)
    return data


def load_df(csv_path):
    
    json_vars = ['device', 'geoNetwork', 'totals', 'trafficSource']  # 'hits' will be handled seperately
    
    # final_vars taken directly from the end of Part 3
    final_vars = ['hits_page.pageTitle_hitssum' , 'hits_hitNumber' , 'hits_referer_hitssum' , 
              'geoNetwork_country_hitssum',  'device_operatingSystem' , 
              'geoNetwork_networkDomain_viewssum' , 'hits_page.pagePathLevel1_hitssum' ,  
              'hits_page.pagePathLevel3_count' , 'hits_page.pageTitle_count' , 'hits_referer_count' ,
              'hits_hour' , 'geoNetwork_country_count' , 'week' , 'trafficSource_referralPath' ,
              'hits_appInfo.landingScreenName_hitssum' , 'hits_appInfo.exitScreenName_hitssum' ,
              'hits_page.pagePathLevel4_count' , 'hits_appInfo.landingScreenName_count' , 
              'geoNetwork_city_hitssum' , 'hour' , 'hits_page.pagePathLevel1_count' ,
              'hits_appInfo.exitScreenName_count' , 'geoNetwork_metro_count' , 'day' ,
              'geoNetwork_networkDomain_hitssum' , 'totals_pageviews' , 'totals_hits' ,
              'geoNetwork_city_count' , 'count' , 'geoNetwork_networkDomain_count' ,
              'totals_timeOnSite' , 'visitNumber' , 'totals_transactionRevenue' , 'fullVisitorId' ]
    
    # the column from the original data we need to import to make the above
    usecols = ['hits' , 'geoNetwork' , 'device' , 'date' , 'trafficSource' , 'visitStartTime' ,
           'totals' , 'visitNumber' , 'fullVisitorId']
    
    print('created json_var, final_var, and usecols')
    
    # lets append json_vars with final_vars, because we still need to import the json vars before expanding them
    all_vars  = json_vars + final_vars + usecols # the master list of columns to import
    
    
    ans = pd.DataFrame()
    
    dfs = pd.read_csv(csv_path, sep=',',
                      converters={column: json.loads for column in json_vars},
                      dtype={'fullVisitorId': 'str'}, # Important!!
                      usecols = usecols,   # import only the ones we really need
                      chunksize = 50000, # 100 000
                      nrows=1000000  # TODO: remove this !
                     )
                        # if memory runs out, try decrease chunksize
    
    for df in dfs:
        df.reset_index(drop = True,inplace = True)
        
        device_list=df['device'].tolist()
        #deleting unwanted columns before normalizing
        for device in device_list:
            del device['browserVersion'],device['browserSize'],device['flashVersion'],device['mobileInputSelector'],device['operatingSystemVersion'],device['screenResolution'],device['screenColors']
        df['device']=pd.Series(device_list)
        
        geoNetwork_list=df['geoNetwork'].tolist()
        for network in geoNetwork_list:
            del network['latitude'],network['longitude'],network['networkLocation'],network['cityId']
        df['geoNetwork']=pd.Series(geoNetwork_list)
        
        df['hits']=df['hits'].apply(ast.literal_eval)
        df['hits']=df['hits'].str[0]
        df['hits']=df['hits'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
        
        # not hits is normal JSON, so we can add it
        json_vars = ['device', 'geoNetwork', 'totals', 'trafficSource', 'hits']  # 'hits' will be handled seperately
    
        
        for column in json_vars:
            column_as_df = pdjson.json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        
        
        df = extraction(df) # FEATURE EXTRACTION
        
        df = cardinality_redux(df) # DEAL WITH HIGH CARDINALITY
        
        df = all_agg(df)
        
        df = encode_num(df)
        
        #df = clean(df)
        
        
        # we wont see each and every column in each chunk that we load, so we need to find where our master list intersects with the actual data
        final_vars = set(final_vars).intersection(df.columns)
        final_vars = list(final_vars)
        df = df[final_vars]
        gc.collect()
        
        df = onehot(df)
        
        ans = pd.concat([ans, df], axis = 0).reset_index(drop = True)
        del(df)
        print('Stored shape:', ans.shape)
        
    return ans

print(" The 'load_df' function has been created")
data1 = load_df('../input/train_v2.csv')
print('data1 shape: ', data1.shape)
print("data1 loaded")
final_vars = ['hits_page.pageTitle_hitssum' , 'hits_hitNumber' , 'hits_referer_hitssum' , 
              'geoNetwork_country_hitssum',  'device_operatingSystem' , 
              'geoNetwork_networkDomain_viewssum' , 'hits_page.pagePathLevel1_hitssum' ,  
              'hits_page.pagePathLevel3_count' , 'hits_page.pageTitle_count' , 'hits_referer_count' ,
              'hits_hour' , 'geoNetwork_country_count' , 'week' , 'trafficSource_referralPath' ,
              'hits_appInfo.landingScreenName_hitssum' , 'hits_appInfo.exitScreenName_hitssum' ,
              'hits_page.pagePathLevel4_count' , 'hits_appInfo.landingScreenName_count' , 
              'geoNetwork_city_hitssum' , 'hour' , 'hits_page.pagePathLevel1_count' ,
              'hits_appInfo.exitScreenName_count' , 'geoNetwork_metro_count' , 'day' ,
              'geoNetwork_networkDomain_hitssum' , 'totals_pageviews' , 'totals_hits' ,
              'geoNetwork_city_count' , 'count' , 'geoNetwork_networkDomain_count' ,
              'totals_timeOnSite' , 'visitNumber' , 'totals_transactionRevenue' , 'fullVisitorId' ]

print('List of Columns not successully loaded: ',list( set(final_vars) - set(data1.columns) ) )

def makeSet(count, data, verbose=1):
    
    # PART 1 ----- Get targets --------
    
    targets = data[ (data['count']>=count+14) & (data['count']<=count+17)][['fullVisitorId', 'totals_transactionRevenue']]
    targets['revenue'] = targets['totals_transactionRevenue']
    targets = targets.drop(labels=['totals_transactionRevenue'], axis=1)
    
    targets['fullVisitorId'] = targets.fullVisitorId.astype('str')
    targets = targets.groupby('fullVisitorId').sum()
    
    targets['fullVisitorId'] = targets.index
    targets.reset_index(drop=True, inplace=True)
    
    # PART 2 ----- Fill in train set --------
    
    train = data[ (data['count']>=count) & (data['count']<=count+10)]
    train=train.copy()
    train['revenue'] = 0 # set all to 0 for now
    
    loyals = targets[targets.revenue>0]['fullVisitorId']
    
    if verbose:
        print(loyals.shape[0], 'buyers in blue-box')
        print( len( list(set(train.fullVisitorId.unique()) & set(targets.fullVisitorId.unique())) ) ,'customers in BOTH green and blue boxes')
        print( len( set(train.fullVisitorId.unique()) & set( loyals ) ) ,'customers return to make a purchase')
    
    loyal_purchaser = set(train.fullVisitorId.unique()) & set( loyals )
    for loyal in loyal_purchaser:
        train.loc[train.fullVisitorId==loyal,'revenue'] = targets[targets.fullVisitorId==loyal]['revenue'].values[0]
    
    revenue=train['revenue']
    train.drop(labels=['revenue'], axis=1, inplace=True)
    return train, revenue

print("The 'makeSet' function has been created")
data2 = load_df('../input/test_v2.csv')
print('data2 shape: ', data2.shape)
print("data2 loaded")

data=data1
del(data1)
data = data.append(data2)
del(data2)

data['totals_transactionRevenue'].fillna(0, inplace=True)
data['totals_transactionRevenue'] = np.log1p(data['totals_transactionRevenue'].astype(float))

data['fullVisitorId'] = data['fullVisitorId'].astype('category',copy=False)

print('data shape: ', data.shape)
def rmse(y_true, y_pred):
    return np.sqrt(sklm.mean_squared_error(y_true, y_pred))


def main():
    importances = pd.DataFrame()
    feature_name = data.columns
    params = { 'metric': 'rmse' }
    est_lgbm = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=32, max_depth=5,
                                  learning_rate=0.01, n_estimators=10000, subsample=0.8, 
                                  subsample_freq=1, colsample_bytree=0.8,
                                  reg_alpha=0.05, reg_lambda=0.05, random_state=1, 
                                  n_jobs = -1, **params)
    
    
    trn_x, trn_y = makeSet(1, data, 0)
    for count in range(2,19):          # (2,37) will range from 2 to 36
        x, y = makeSet(count, data, 0)
        trn_x=trn_x.append(x)
        trn_y=trn_y.append(y)
        del(x,y)
    #
    
    val_x, val_y = makeSet(43, data, 0)  # 43
    # val_y will be all zeros
    
    # Train estimator.
    est_lgbm.fit(trn_x, trn_y,
                 eval_set=[(val_x, val_y)],
                 early_stopping_rounds=50, 
                 verbose=False)
    # Prediction and evaluation on validation data set.
    val_pred = est_lgbm.predict(val_x)
    rmse_valid = rmse(val_y, np.maximum(0, val_pred))
    
    gc.collect()
    
    fullVisitorId = val_x['fullVisitorId']
    
    return fullVisitorId, val_pred, rmse_valid
    
if __name__ == '__main__':
    fullVisitorId, val_pred, rmse_valid = main()
#
print('done')
fullVisitorId = pd.DataFrame(fullVisitorId)
fullVisitorId = fullVisitorId.reset_index(drop=True)
val_pred = pd.DataFrame(val_pred)

submission = fullVisitorId.copy()
submission['PredictedLogRevenue'] = val_pred

del(fullVisitorId, val_pred)

print('Will there be any buyers in Dec 2018 or Jan 2019?', (submission.PredictedLogRevenue>0).any())

#print('done')
print( submission.shape)
submission.drop_duplicates(subset='fullVisitorId', inplace=True)
print( submission.shape)
#submit file
submission.to_csv("../working/submission.csv", index=False)

print("Submitted to 'Output'")