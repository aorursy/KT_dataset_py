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
json_vars = ['device', 'geoNetwork', 'totals', 'trafficSource', 'hits', 'customDimensions']

# final_vars taken directly from the end of Part 2
final_vars = ['fullVisitorId', 'month','week','weekday','hour', 'year', 'day','totals_transactionRevenue']
print('created json_var and final_var')


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
    df['weekday'] = df['date'].dt.weekday
    # Get the year
    df['year'] = df['date'].dt.year
    # Get the day of the month
    df['day'] = df['date'].dt.day
    df = df.drop(labels=['date'], axis=1)
    
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['hour'] = df['visitStartTime'].dt.hour
    df = df.drop(labels=['visitStartTime'], axis=1)
    
    return df


def load_df(csv_path, usecols=None):
    JSON_COLUMNS = ['totals']#['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    
    dfs = pd.read_csv(csv_path, sep=',',
                      converters={column: json.loads for column in JSON_COLUMNS},
                      dtype={'fullVisitorId': 'str'}, # Important!!
                      chunksize = 100000, # 100 000
                      #nrows=2000,  # TODO: remove this !
                      usecols=['fullVisitorId', 'date','visitStartTime', 'totals']
                     )
                        # if memory runs out, try decrease chunksize
    
    for df in dfs:
        df.reset_index(drop = True,inplace = True)
        
        for column in JSON_COLUMNS:
            column_as_df = pdjson.json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        
        
        df = extraction(df) # FEATURE EXTRACTION
        
        
        # we wont see each and every column in each chunk that we load, so we need to find where our master list intersects with the actual data
        usecols = set(usecols).intersection(df.columns)
        usecols = list(usecols)
        use_df = df[usecols]
        del df
        
        #use_df = cardinality_redux(use_df) # DEAL WITH HIGH CARDINALITY
        
        gc.collect()
        ans = pd.concat([ans, use_df], axis = 0).reset_index(drop = True)
        print('Stored shape:', ans.shape)
        
    return ans

data1 = load_df('../input/train_v2.csv', usecols=final_vars)
print('data1 shape: ', data1.shape)
print("data1 loaded")

data1.head()
data1['totals_transactionRevenue'].fillna(0, inplace=True)
data1['totals_transactionRevenue'] = np.log1p(data1['totals_transactionRevenue'].astype(float))

data2 = load_df('../input/test_v2.csv', usecols=final_vars)
print('data2 shape: ', data2.shape)
print("data2 loaded")
data2['totals_transactionRevenue'].fillna(0, inplace=True)
data2['totals_transactionRevenue'] = np.log1p(data2['totals_transactionRevenue'].astype(float))

data2.head()
# we take 1 through 15th of each month to be 1 period
# the 16th till end is period 2

data1['count'] = ((data1['year'] - 2016) * 24) + (data1['month']-1)*2 + round( data1['day']/30 ) - 13

data1[['year', 'month', 'day', 'count']].sort_values('count').head()

# we take 1 through 15th of each month to be 1 period
# the 16th till end is period 2

data2['count'] = ((data2['year'] - 2016) * 24) + (data2['month']-1)*2 + round( data2['day']/30 ) - 13

data2[['year', 'month', 'day', 'count']].sort_values('count').head()

print('Range of periods in train set:\n',
      data1['count'].min(), 'min\n',
      data1['count'].max(), 'max\n',
     )

print('Range of periods in test set:\n',
      data2['count'].min(), 'min\n',
      data2['count'].max(), 'max\n',
     )
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

print('done')

data = data1.append(data2)
del(data1, data2)

for count in range(1,37):   # epoch 1 through 36
    print('for epoch', count, '----------------------------------------')
    train,revenue = makeSet(count,data)
    print( 'number of visits overwritten with a future revenue:', (revenue>0).sum() )
    del(train)

json_vars = ['device', 'geoNetwork', 'totals', 'trafficSource', 'hits', 'customDimensions']

# final_vars taken directly from the end of Part 2
final_vars = ['channelGrouping','customDimensions_index','customDimensions_value','device_browser',
'device_deviceCategory','device_operatingSystem','fullVisitorId','geoNetwork_city',
'geoNetwork_continent','geoNetwork_country','geoNetwork_metro','geoNetwork_networkDomain',
'geoNetwork_region','geoNetwork_subContinent','hits_appInfo.exitScreenName',
'hits_appInfo.landingScreenName','hits_appInfo.screenName','hits_contentGroup.contentGroup2',
'hits_contentGroup.contentGroupUniqueViews2','hits_dataSource','hits_eCommerceAction.option',
'hits_eventInfo.eventCategory','hits_eventInfo.eventLabel','hits_hitNumber','hits_hour',
'hits_isEntrance','hits_isExit','hits_item.currencyCode','hits_latencyTracking.domContentLoadedTime',
'hits_latencyTracking.domInteractiveTime','hits_latencyTracking.domLatencyMetricsSample',
'hits_latencyTracking.domainLookupTime','hits_latencyTracking.pageDownloadTime',
'hits_latencyTracking.pageLoadSample','hits_latencyTracking.pageLoadTime',
'hits_latencyTracking.redirectionTime','hits_latencyTracking.serverConnectionTime',
'hits_latencyTracking.serverResponseTime','hits_latencyTracking.speedMetricsSample',
'hits_page.hostname','hits_page.pagePath','hits_page.pagePathLevel1','hits_page.pagePathLevel2',
'hits_page.pagePathLevel3','hits_page.pagePathLevel4','hits_page.pageTitle',
'hits_promotionActionInfo.promoIsView','hits_referer','hits_social.hasSocialSourceReferral',
'hits_social.socialNetwork','hits_transaction.currencyCode','hits_type','totals_bounces',
'totals_hits','totals_newVisits','totals_pageviews','totals_sessionQualityDim','totals_timeOnSite',
'totals_transactionRevenue','totals_transactions','trafficSource_adContent',
'trafficSource_adwordsClickInfo.adNetworkType','trafficSource_adwordsClickInfo.gclId',
'trafficSource_adwordsClickInfo.isVideoAd','trafficSource_adwordsClickInfo.slot',
'trafficSource_isTrueDirect','trafficSource_medium','trafficSource_referralPath',
'trafficSource_source','visitId','visitNumber','month','week','weekday','geoNetwork_city_count',
'geoNetwork_city_hitssum','geoNetwork_city_viewssum','hour', 'year', 'day']
# added year to this list
# added day to this list

print('created json_var and final_var')


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
    df['weekday'] = df['date'].dt.weekday
    # Get the year
    df['year'] = df['date'].dt.year
    # Get the day of the month
    df['day'] = df['date'].dt.day
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
           'hits_appInfo.landingScreenName', 'hits_appInfo.screenName', 'hits_eventInfo.eventLabel', 
           'hits_page.pagePath', 'hits_page.pagePathLevel1', 'hits_page.pagePathLevel2', 
           'hits_page.pagePathLevel3', 'hits_page.pagePathLevel4', 'hits_page.pageTitle', 
           'hits_referer', 'trafficSource_adContent', 'trafficSource_adwordsClickInfo.gclId']
    for dfstr in lst:
        df = make_countsum(df, dfstr)
    return df

# lets append json_vars with final_vars, because we still need to import the json vars before expanding them
all_vars  = json_vars + final_vars # the master list of columns to import

def load_df(csv_path, usecols=None):
    JSON_COLUMNS = ['totals','device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    
    dfs = pd.read_csv(csv_path, sep=',',
                      converters={column: json.loads for column in JSON_COLUMNS},
                      dtype={'fullVisitorId': 'str'}, # Important!!
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
    
        df['customDimensions']=df['customDimensions'].apply(ast.literal_eval)
        df['customDimensions']=df['customDimensions'].str[0]
        df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
        JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource','hits','customDimensions']

        for column in JSON_COLUMNS:
            column_as_df = pdjson.json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        
        
        df = extraction(df) # FEATURE EXTRACTION
        
        
        # we wont see each and every column in each chunk that we load, so we need to find where our master list intersects with the actual data
        usecols = set(usecols).intersection(df.columns)
        usecols = list(usecols)
        use_df = df[usecols]
        del df
        
        use_df = cardinality_redux(use_df) # DEAL WITH HIGH CARDINALITY
        
        gc.collect()
        ans = pd.concat([ans, use_df], axis = 0).reset_index(drop = True)
        print('Stored shape:', ans.shape)
        
    return ans

data1 = load_df('../input/train_v2.csv', usecols=final_vars)
print('data1 shape: ', data1.shape)
print("data1 loaded")
data1['totals_transactionRevenue'].fillna(0, inplace=True)
data1['totals_transactionRevenue'] = np.log1p(data1['totals_transactionRevenue'].astype(float))

#data2 = load_df('../input/test_v2.csv', usecols=final_vars)
#data2['totals_transactionRevenue'].fillna(0, inplace=True)
#data2['totals_transactionRevenue'] = np.log1p(data2['totals_transactionRevenue'].astype(float))
#print('data2 shape: ', data2.shape)
#print("data2 loaded")

#test_target = data2['totals_transactionRevenue']
#data2.drop(labels=['totals_transactionRevenue'], axis=1, inplace=True) # remove targets

#data1 = data1.append(data2)  # Append rows of data2 to data1
#del(data2)
data=data1
del(data1)
print('shape of data:', data.shape)

data.head()
def aggregate(df, col, leave): # fn to aggregate all categories in df[col] except for cols in leave
    df[col] = df[col].astype('str')
    include = df[col].unique()  # array of all unique categories
    include = list(include)
    include = set(include).difference(set(leave))  # set: take out 'leave' from include
    include = list(include)
    df.loc[df[col].isin(include), col] = "grouped"  # rename all cols in 'include' to 'grouped'
    return df


data = aggregate(data, 'customDimensions_value', leave=['North America'])
data = aggregate(data, 'device_browser', leave=['Chrome', 'Safari', 'Firefox', 'Internet Explorer'])
data = aggregate(data, 'device_operatingSystem', leave=['Windows', 'Macintosh', 'Android', 'iOS', 'Linux', 'Chrome OS'])
data = aggregate(data, 'geoNetwork_continent', leave=['Americas'])
data = aggregate(data, 'geoNetwork_subContinent', leave=['Northern America'])
data = aggregate(data, 'hits_contentGroup.contentGroup2', leave=['(not set)', 'Brands', 'Apparel'])
data = aggregate(data, 'hits_social.socialNetwork', leave=['(not set)', 'YouTube'])
data = aggregate(data, 'totals_transactions', leave=['nan'])
data = aggregate(data, 'trafficSource_referralPath', leave=['/'])
data = aggregate(data, 'trafficSource_source', leave=['(direct)','google','youtube.com'])

print('done')
for col in data.columns:
    x=data[col]
    if (x.apply(np.isreal).all(axis=0)) & ((str(x.dtypes) != 'category')): # if numeric, but not category
        #print(col, 'is numeric')
        1+1
    else:
        print(col, 'has', data[col].nunique(dropna=False), 'categories')
data['hits_hitNumber'] = pd.to_numeric(data['hits_hitNumber'], errors='coerce', downcast='unsigned')
data['hits_hour'] = pd.to_numeric(data['hits_hour'], errors='coerce', downcast='unsigned')
data['totals_sessionQualityDim'] = pd.to_numeric(data['totals_sessionQualityDim'], errors='coerce', downcast='unsigned')
data['totals_timeOnSite'] = pd.to_numeric(data['totals_timeOnSite'], errors='coerce', downcast='unsigned')
print('done')
print('Showing numeric features with nans only:')
for col in data.columns:
    if (data[col].isnull().any()) & (col != 'fullVisitorId') & (col != 'visitId'): # keep IDs as string
        if (data[col].apply(np.isreal).all(axis=0)) & ((str(data[col].dtypes) != 'category')):
            print(col, 'has', data[col].isnull().sum(), 'nans' ) # check each column.
print('------------------------------------------------------------------------------')
print('Showing categorical features with nans only:')
for col in data.columns:
    if (data[col].isnull().any()):
        if (data[col].apply(np.isreal).all(axis=0))==0 & ((str(data[col].dtypes) != 'category'))==0:
            print(col, 'has', data[col].isnull().sum(), 'nans' ) # check each column.

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

print('done')
print('Showing numeric features with nans only:')
for col in data.columns:
    if (data[col].isnull().any()):
        if (data[col].apply(np.isreal).all(axis=0)) & ((str(data[col].dtypes) != 'category')):
            print(col, 'has', data[col].isnull().sum(), 'nans' ) # check each column.
print('------------------------------------------------------------------------------')
print('Showing categorical features only:')
for col in data.columns:
    x=data[col]
    if (x.apply(np.isreal).all(axis=0)) & ((str(x.dtypes) != 'category')): # if numeric, but not category
        #print(col, 'is numeric')
        1+1
    elif data[col].nunique(dropna=False) > 2:
        print(col, 'has', data[col].nunique(dropna=False), 'categories')
    elif data[col].nunique(dropna=False) == 1:
        print(col, 'has', data[col].nunique(dropna=False), 'categories <-----------------')

for col in data.columns:
    x=data[col]
    if (x.apply(np.isreal).all(axis=0)) & ((str(x.dtypes) != 'category')): # if numeric, but not category
        #print(col, 'is numeric')
        1+1
    else:
        data[col] = data[col].astype('category',copy=False)
print('done')
data.dtypes
# we take the 1st through 15th of each month to be 1 period
# the 16th till end is period 2

data['count'] = ((data['year'] - 2016) * 24) + (data['month']-1)*2 + round( data['day']/30 ) - 13

data[['year', 'month', 'day', 'count']].sort_values('count').head()

print('done')
data.head()
data.drop(labels=['visitId','year'], axis=1, inplace=True)
print('done')
def rmse(y_true, y_pred):
    return np.sqrt(sklm.mean_squared_error(y_true, y_pred))


def plot_importances(imps):
    mean_gain = np.log1p(imps[['gain', 'feature']].groupby('feature').mean())
    imps['mean_gain'] = imps['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(10, 16))
    sns.barplot(x='gain', y='feature', data=imps.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    #plt.savefig('importances.png')




def main():
    importances = pd.DataFrame()
    feature_name = data.columns
    params = { 'metric': 'rmse' }
    est_lgbm = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=32, max_depth=5,
                                  learning_rate=0.01, n_estimators=10000, subsample=0.8, 
                                  subsample_freq=1, colsample_bytree=0.8,
                                  reg_alpha=0.05, reg_lambda=0.05, random_state=1, **params)
    
    
    for count in range(1,11): # (1,11) will range from 1 to 10
        trn_x, trn_y = makeSet(count, data, 0)
        x, y = makeSet(count+1, data, 0)
        trn_x=trn_x.append(x)
        trn_y=trn_y.append(y)
        trn_x, trn_y = makeSet(count+2, data, 0)
        trn_x=trn_x.append(x)
        trn_y=trn_y.append(y)
        del(x,y)
        val_x, val_y = makeSet(count+3, data, 0) # give each set a chance to be the validation set
        
        # Train estimator.
        est_lgbm.fit(trn_x, trn_y, 
                     eval_set=[(val_x, val_y)],
                     eval_metric='rmse',
                     early_stopping_rounds=50, 
                     verbose=False)
        # Prediction and evaluation on validation data set.
        val_pred = est_lgbm.predict(val_x)
        rmse_valid = rmse(val_y, np.maximum(0, val_pred))
        #mean_rmse += rmse_valid
        #print("%d RMSE: %.5f" % (fold_idx + 1, rmse_valid))
        # Prediction of testing data set.
        #y_test_vec += np.expm1(np.maximum(0, est_lgbm.predict(test_df)))
        # Set feature importances.
        imp_df = pd.DataFrame()
        imp_df['feature'] = feature_name
        imp_df['gain'] = est_lgbm.feature_importances_
        #imp_df['fold'] = fold_idx + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
        
        #print("Mean RMSE: %.5f" % (mean_rmse / N_SPLITS))
        
        #del train_df, test_df, train_tvals
        gc.collect()
        
        #imps = imps.
    
    # Plot feature importances
    print('Plot feature importances')
    #print(importances)
    plot_importances(importances)
    
    
if __name__ == '__main__':
    main()