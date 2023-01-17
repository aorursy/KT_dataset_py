import gc
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
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype(float)
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0)
train_df['totals.transactionRevenue'] = np.log1p(train_df['totals.transactionRevenue'])
cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
train_df.drop(['trafficSource.campaignCode'], axis=1, inplace=True)
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
    train_df[col] = np.log1p((train_df[col].values)).astype('float32')
    test_df[col] = np.log1p((test_df[col].values)).astype('float32')
test_df['totals.transactionRevenue'] = -1
for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str'))).astype('float32')
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str'))).astype('float32')
train_df.fillna(-1,inplace=True,axis=1)
test_df.fillna(-1,inplace=True,axis=1)
df = pd.concat([train_df,test_df],sort=False)
df = df.reset_index(drop=True)
for col in num_cols:
    df.loc[:,col] = pd.cut(df[col], 50,labels=False)
gp_test_df = df[df['totals.transactionRevenue']==-1].copy().reset_index(drop=True)
gp_train_df = df[df['totals.transactionRevenue']!=-1].copy().reset_index(drop=True)
del df
gc.collect()
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
features = num_cols+cat_cols
for c in features:
    print(c)
    tr,te =target_encode(gp_train_df[c], 
                         gp_test_df[c], 
                         target=(gp_train_df['totals.transactionRevenue']>0).astype(int), 
                         min_samples_leaf=50,
                         smoothing=0,
                         noise_level=0.0)
    gp_train_df[c] = tr
    gp_test_df[c] = te
df = pd.concat([gp_train_df,gp_test_df],sort=False)
del gp_train_df
del gp_test_df
gc.collect()
from sklearn.preprocessing import StandardScaler
for c in features:
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
    df[c].fillna(-99999,inplace=True)
gp_test_df = df[df['totals.transactionRevenue']==-1].copy().reset_index(drop=True)
gp_train_df = df[df['totals.transactionRevenue']!=-1].copy().reset_index(drop=True)
del df
gc.collect()
gp_test_df.drop('totals.transactionRevenue',axis=1,inplace=True)
test_df.drop('totals.transactionRevenue',axis=1,inplace=True)
def Output(p):
    return 1.0/(1.0+np.exp(-p))

def GP1(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh(np.minimum(((3.0)), ((data["totals.pageviews"])))) 
    v["i1"] = 0.100000*np.tanh((((data["geoNetwork.region"]) > (-1.0))*1.)) 
    v["i2"] = 0.100000*np.tanh(np.minimum(((np.maximum(((data["mean.hits.per.network.domain"])), ((3.0))))), ((3.0)))) 
    v["i3"] = 0.100000*np.tanh(((2.0) + (data["geoNetwork.region"]))) 
    v["i4"] = 0.100000*np.tanh(((-1.0) * (((data["totals.pageviews"]) * ((14.40841960906982422)))))) 
    v["i5"] = 0.100000*np.tanh(((((((1.0) < (-2.0))*1.)) < (((data["geoNetwork.subContinent"]) - (data["sum.pageviews.per.network.domain"]))))*1.)) 
    v["i6"] = 0.100000*np.tanh(np.maximum((((((data["totals.hits"]) + (data["totals.pageviews"]))/2.0))), ((data["totals.hits"])))) 
    v["i7"] = 0.100000*np.tanh(((2.0) * (data["totals.pageviews"]))) 
    v["i8"] = 0.100000*np.tanh(((0.0) - (data["totals.pageviews"]))) 
    v["i9"] = 0.100000*np.tanh(np.maximum(((0.0)), ((data["geoNetwork.continent"])))) 
    v["i10"] = 0.100000*np.tanh(((-1.0) * (data["totals.bounces"]))) 
    v["i11"] = 0.100000*np.tanh((((-3.0) < (((3.0) - ((7.0)))))*1.)) 
    v["i12"] = 0.100000*np.tanh(((np.maximum(((data["mean.pageviews.per.network.domain"])), (((((-3.0) > (data["totals.hits"]))*1.))))) - (data["totals.hits"]))) 
    v["i13"] = 0.100000*np.tanh((((-1.0*((-1.0)))) / 2.0)) 
    v["i14"] = 0.100000*np.tanh((((np.maximum(((-1.0)), ((np.maximum(((data["browser.category"])), ((1.0))))))) > ((-1.0*((data["totals.pageviews"])))))*1.)) 
    v["i15"] = 0.100000*np.tanh(np.minimum(((data["totals.pageviews"])), ((data["source.country"])))) 
    v["i16"] = 0.100000*np.tanh(((((1.0)) > (data["trafficSource.isTrueDirect"]))*1.)) 
    v["i17"] = 0.100000*np.tanh((((data["geoNetwork.region"]) + (data["totals.pageviews"]))/2.0)) 
    v["i18"] = 0.100000*np.tanh((((data["mean.hits.per.network.domain"]) + ((((data["totals.pageviews"]) > (3.0))*1.)))/2.0)) 
    v["i19"] = 0.100000*np.tanh((((data["sum.hits.per.network.domain"]) < ((((9.0)) + (data["geoNetwork.city"]))))*1.)) 
    v["i20"] = 0.100000*np.tanh(np.minimum(((data["totals.pageviews"])), (((((data["visitNumber"]) + (((data["source.country"]) * (data["totals.pageviews"]))))/2.0))))) 
    v["i21"] = 0.100000*np.tanh((((((((((data["totals.pageviews"]) + (-1.0))/2.0)) + (((data["totals.pageviews"]) + (data["totals.pageviews"]))))/2.0)) + ((((data["totals.pageviews"]) + (data["totals.bounces"]))/2.0)))/2.0)) 
    v["i22"] = 0.100000*np.tanh((((data["trafficSource.source"]) < ((((np.tanh((data["mean.pageviews.per.network.domain"]))) < ((((data["geoNetwork.subContinent"]) < ((-1.0*((data["geoNetwork.networkDomain"])))))*1.)))*1.)))*1.)) 
    v["i23"] = 0.100000*np.tanh(np.minimum(((data["totals.pageviews"])), ((data["geoNetwork.subContinent"])))) 
    v["i24"] = 0.100000*np.tanh(np.maximum(((-3.0)), ((np.where(((data["totals.pageviews"]) * 2.0)<0, (1.0), -3.0 ))))) 
    v["i25"] = 0.100000*np.tanh((((((((data["totals.pageviews"]) - (data["totals.hits"]))) + (data["totals.pageviews"]))/2.0)) / 2.0)) 
    v["i26"] = 0.100000*np.tanh(np.where(data["totals.pageviews"]>0, data["source.country"], np.minimum(((data["source.country"])), ((data["totals.pageviews"]))) )) 
    v["i27"] = 0.099951*np.tanh(((np.minimum((((((data["totals.pageviews"]) + (data["totals.pageviews"]))/2.0))), ((data["source.country"])))) + (data["totals.pageviews"]))) 
    v["i28"] = 0.100000*np.tanh(((((np.minimum(((data["source.country"])), (((((data["source.country"]) + (data["totals.pageviews"]))/2.0))))) + ((((data["totals.pageviews"]) + (-3.0))/2.0)))) * 2.0)) 
    v["i29"] = 0.100000*np.tanh((((((data["totals.pageviews"]) * 2.0)) < ((((data["medium.hits.sum"]) < (data["totals.hits"]))*1.)))*1.)) 
    v["i30"] = 0.100000*np.tanh(np.tanh((((np.where(-1.0>0, 1.0, data["totals.hits"] )) * 2.0)))) 
    v["i31"] = 0.100000*np.tanh(((((((data["totals.pageviews"]) + (((data["trafficSource.source"]) - (3.0))))/2.0)) + (((data["totals.pageviews"]) - (data["totals.hits"]))))/2.0)) 
    v["i32"] = 0.100000*np.tanh(((((((np.minimum(((data["totals.pageviews"])), ((data["totals.newVisits"])))) * 2.0)) * 2.0)) + ((((data["totals.pageviews"]) + (np.minimum(((data["totals.newVisits"])), ((data["totals.newVisits"])))))/2.0)))) 
    v["i33"] = 0.100000*np.tanh(((((np.minimum(((data["totals.hits"])), ((((data["browser.os"]) * 2.0))))) * 2.0)) + ((((data["geoNetwork.networkDomain"]) + (data["totals.pageviews"]))/2.0)))) 
    v["i34"] = 0.100000*np.tanh((((-1.0*((np.minimum(((data["source.country"])), ((((((data["totals.pageviews"]) / 2.0)) + (-2.0))))))))) - (data["visitNumber"]))) 
    v["i35"] = 0.100000*np.tanh(((np.minimum(((data["totals.pageviews"])), ((data["totals.pageviews"])))) + (np.where(((np.minimum(((data["totals.bounces"])), ((data["totals.newVisits"])))) * 2.0)>0, data["totals.bounces"], -3.0 )))) 
    v["i36"] = 0.100000*np.tanh(np.where(np.where((-1.0*((data["geoNetwork.country"]))) < -99998, data["geoNetwork.metro"], data["geoNetwork.country"] )>0, (-1.0*((data["totals.bounces"]))), (-1.0*((data["geoNetwork.city"]))) )) 
    v["i37"] = 0.100000*np.tanh(((np.minimum(((data["device.deviceCategory"])), ((data["source.country"])))) + (((((np.minimum(((data["device.deviceCategory"])), ((((data["totals.pageviews"]) * 2.0))))) * 2.0)) * 2.0)))) 
    v["i38"] = 0.100000*np.tanh(np.minimum(((data["totals.newVisits"])), ((data["totals.newVisits"])))) 
    v["i39"] = 0.100000*np.tanh(((((((np.tanh(((-1.0*((data["browser.os"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i40"] = 0.100000*np.tanh((((((((((data["totals.hits"]) / 2.0)) < (data["totals.pageviews"]))*1.)) + (data["source.country"]))) * 2.0)) 
    v["i41"] = 0.100000*np.tanh(((data["totals.hits"]) - (data["totals.pageviews"]))) 
    v["i42"] = 0.100000*np.tanh(((((data["totals.pageviews"]) - (data["totals.hits"]))) * (data["totals.hits"]))) 
    v["i43"] = 0.100000*np.tanh(((np.minimum(((data["month.unique.user.count"])), ((data["source.country"])))) + (((np.minimum(((data["month.unique.user.count"])), ((((np.minimum(((data["month.unique.user.count"])), ((data["totals.pageviews"])))) * 2.0))))) * 2.0)))) 
    v["i44"] = 0.100000*np.tanh(np.tanh(((-1.0*((data["totals.hits"])))))) 
    v["i45"] = 0.100000*np.tanh(np.minimum(((data["device.deviceCategory"])), ((np.where(((data["device.deviceCategory"]) * 2.0)>0, data["totals.pageviews"], data["totals.pageviews"] ))))) 
    v["i46"] = 0.100000*np.tanh(((data["device.operatingSystem"]) + (np.where(data["geoNetwork.city"]>0, ((data["browser.os"]) + (data["trafficSource.source"])), ((data["trafficSource.source"]) + (data["geoNetwork.country"])) )))) 
    v["i47"] = 0.100000*np.tanh((((((data["geoNetwork.country"]) + (data["totals.pageviews"]))) + (data["totals.bounces"]))/2.0)) 
    v["i48"] = 0.100000*np.tanh(((data["totals.hits"]) - ((4.81635427474975586)))) 
    v["i49"] = 0.100000*np.tanh(((np.minimum(((data["month.unique.user.count"])), ((data["totals.pageviews"])))) * 2.0)) 
    v["i50"] = 0.097362*np.tanh((((data["geoNetwork.metro"]) < (((0.0) * (data["geoNetwork.city"]))))*1.)) 
    v["i51"] = 0.100000*np.tanh(((((((((((((((data["geoNetwork.subContinent"]) * (data["geoNetwork.region"]))) * 2.0)) * (data["totals.pageviews"]))) * 2.0)) * 2.0)) * (data["totals.pageviews"]))) * 2.0)) 
    v["i52"] = 0.100000*np.tanh(((((np.where(data["totals.pageviews"]<0, data["totals.pageviews"], data["day"] )) * 2.0)) * 2.0)) 
    v["i53"] = 0.100000*np.tanh((((((((data["totals.pageviews"]) + (data["mean.pageviews.per.network.domain"]))) < (data["source.country"]))*1.)) * (((((((data["browser.category"]) + (data["mean.pageviews.per.network.domain"]))/2.0)) > (3.0))*1.)))) 
    v["i54"] = 0.100000*np.tanh((((((((((data["browser.os"]) > (data["totals.hits"]))*1.)) > (((((9.0)) < (data["channelGrouping"]))*1.)))*1.)) < ((((data["totals.pageviews"]) < (data["trafficSource.source"]))*1.)))*1.)) 
    v["i55"] = 0.099951*np.tanh((((-2.0) + (data["device.deviceCategory"]))/2.0)) 
    v["i56"] = 0.099951*np.tanh(((np.tanh((data["totals.hits"]))) + (((((data["totals.hits"]) - (data["totals.pageviews"]))) + (data["totals.hits"]))))) 
    v["i57"] = 0.100000*np.tanh(((((((((data["trafficSource.campaign"]) + (data["count.hits.per.network.domain"]))) + (data["trafficSource.campaign"]))/2.0)) < (data["geoNetwork.continent"]))*1.)) 
    v["i58"] = 0.099902*np.tanh(((((np.minimum(((np.minimum(((data["weekday"])), ((data["totals.bounces"]))))), ((data["totals.bounces"])))) * 2.0)) * 2.0)) 
    v["i59"] = 0.100000*np.tanh(((data["geoNetwork.country"]) * (data["totals.hits"]))) 
    v["i60"] = 0.100000*np.tanh((((-3.0) + (((((((-3.0) + (data["geoNetwork.city"]))) + ((((data["totals.pageviews"]) + (data["geoNetwork.city"]))/2.0)))) + (data["totals.pageviews"]))))/2.0)) 
    v["i61"] = 0.100000*np.tanh(((-2.0) * (((data["totals.pageviews"]) - (data["totals.hits"])))))
    return v.sum(axis=1)

def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.100000*np.tanh(((((-1.0*((data["device.isMobile"])))) < (np.where(-1.0 < -99998, data["totals.hits"], (((-1.0*((2.0)))) - (3.0)) )))*1.)) 
    v["i1"] = 0.100000*np.tanh((((data["totals.pageviews"]) > (-3.0))*1.)) 
    v["i2"] = 0.100000*np.tanh(((np.where(np.minimum(((2.0)), ((data["geoNetwork.region"])))<0, np.maximum(((data["totals.hits"])), ((-2.0))), 2.0 )) - (data["mean.hits.per.network.domain"]))) 
    v["i3"] = 0.100000*np.tanh(((3.0) - (data["totals.hits"]))) 
    v["i4"] = 0.100000*np.tanh(((data["totals.pageviews"]) * ((6.41743326187133789)))) 
    v["i5"] = 0.100000*np.tanh(np.minimum(((2.0)), ((data["geoNetwork.country"])))) 
    v["i6"] = 0.100000*np.tanh(((data["totals.pageviews"]) * ((3.0)))) 
    v["i7"] = 0.100000*np.tanh((((-2.0) > (np.where(2.0<0, 1.0, (((data["geoNetwork.networkDomain"]) + (data["browser.os"]))/2.0) )))*1.)) 
    v["i8"] = 0.100000*np.tanh((((((0.0) - (data["totals.hits"]))) < (-3.0))*1.)) 
    v["i9"] = 0.100000*np.tanh(((data["totals.pageviews"]) * (data["totals.hits"]))) 
    v["i10"] = 0.100000*np.tanh(((data["geoNetwork.country"]) + (data["totals.hits"]))) 
    v["i11"] = 0.100000*np.tanh(np.where(np.tanh((2.0))<0, (-1.0*((((2.0) / 2.0)))), data["totals.pageviews"] )) 
    v["i12"] = 0.100000*np.tanh(((((((data["totals.hits"]) > (0.0))*1.)) > ((((2.0) < (data["totals.pageviews"]))*1.)))*1.)) 
    v["i13"] = 0.100000*np.tanh(((np.where(data["geoNetwork.continent"]>0, data["totals.pageviews"], (((np.tanh((data["sum.hits.per.network.domain"]))) < (((data["geoNetwork.metro"]) / 2.0)))*1.) )) / 2.0)) 
    v["i14"] = 0.100000*np.tanh(((data["geoNetwork.country"]) + (data["totals.pageviews"]))) 
    v["i15"] = 0.100000*np.tanh((-1.0*((((data["totals.pageviews"]) - (1.0)))))) 
    v["i16"] = 0.100000*np.tanh((((((8.0)) * (np.minimum(((((data["geoNetwork.continent"]) / 2.0))), ((np.minimum(((data["geoNetwork.continent"])), ((data["totals.pageviews"]))))))))) + (((data["geoNetwork.continent"]) / 2.0)))) 
    v["i17"] = 0.100000*np.tanh(np.minimum((((((data["source.country"]) + (np.minimum((((((data["source.country"]) + ((((-1.0) + (data["totals.pageviews"]))/2.0)))/2.0))), ((data["totals.pageviews"])))))/2.0))), ((data["totals.pageviews"])))) 
    v["i18"] = 0.100000*np.tanh((((((np.minimum(((data["geoNetwork.country"])), ((data["totals.pageviews"])))) * 2.0)) + (((np.minimum(((data["geoNetwork.country"])), ((data["device.isMobile"])))) + (data["totals.pageviews"]))))/2.0)) 
    v["i19"] = 0.100000*np.tanh(np.minimum((((1.24994897842407227))), (((((0.0) > ((((-3.0) < (data["trafficSource.source"]))*1.)))*1.))))) 
    v["i20"] = 0.100000*np.tanh(np.maximum(((data["totals.newVisits"])), ((data["totals.hits"])))) 
    v["i21"] = 0.100000*np.tanh((((data["totals.newVisits"]) < ((((2.0) < ((((data["browser.os"]) > (data["visitNumber"]))*1.)))*1.)))*1.)) 
    v["i22"] = 0.100000*np.tanh(np.maximum(((np.minimum(((np.minimum(((data["totals.hits"])), ((data["geoNetwork.subContinent"]))))), ((data["totals.hits"]))))), ((-1.0)))) 
    v["i23"] = 0.100000*np.tanh((((((data["geoNetwork.subContinent"]) * (np.where(data["totals.pageviews"]<0, data["totals.pageviews"], data["totals.pageviews"] )))) + ((((data["totals.pageviews"]) + (data["totals.pageviews"]))/2.0)))/2.0)) 
    v["i24"] = 0.100000*np.tanh(np.minimum((((5.0))), ((data["trafficSource.adwordsClickInfo.page"])))) 
    v["i25"] = 0.100000*np.tanh(np.tanh((np.minimum((((3.0))), ((np.where(2.0<0, data["count.pageviews.per.network.domain"], -1.0 ))))))) 
    v["i26"] = 0.100000*np.tanh(np.where(((data["totals.pageviews"]) * 2.0)<0, -3.0, (((((data["totals.pageviews"]) * 2.0)) + (((data["totals.hits"]) * 2.0)))/2.0) )) 
    v["i27"] = 0.099951*np.tanh(np.maximum(((data["totals.hits"])), ((np.tanh(((((data["totals.hits"]) + (np.minimum(((np.tanh((((1.0) * 2.0))))), ((data["totals.pageviews"])))))/2.0))))))) 
    v["i28"] = 0.100000*np.tanh(np.minimum(((data["source.country"])), ((np.tanh((((((data["totals.pageviews"]) - ((1.56775867938995361)))) + (data["totals.newVisits"])))))))) 
    v["i29"] = 0.100000*np.tanh(np.where(data["totals.pageviews"]>0, ((data["totals.pageviews"]) - ((((data["totals.bounces"]) > (((data["trafficSource.keyword"]) * 2.0)))*1.))), (((data["trafficSource.adwordsClickInfo.isVideoAd"]) < (data["totals.pageviews"]))*1.) )) 
    v["i30"] = 0.100000*np.tanh(((((((data["totals.pageviews"]) + (data["trafficSource.source"]))) + (-3.0))) + (((data["trafficSource.isTrueDirect"]) + (data["trafficSource.isTrueDirect"]))))) 
    v["i31"] = 0.100000*np.tanh(((np.minimum(((np.where(data["geoNetwork.continent"]>0, data["totals.pageviews"], ((data["geoNetwork.region"]) * 2.0) ))), ((((((data["geoNetwork.continent"]) * 2.0)) * 2.0))))) * 2.0)) 
    v["i32"] = 0.100000*np.tanh((((((((((((((-3.0) / 2.0)) < (-2.0))*1.)) < (((-3.0) / 2.0)))*1.)) * 2.0)) > ((13.35829830169677734)))*1.)) 
    v["i33"] = 0.100000*np.tanh(np.minimum(((((np.minimum(((data["totals.pageviews"])), ((data["source.country"])))) + ((((data["totals.pageviews"]) + (((-3.0) + (data["source.country"]))))/2.0))))), ((data["source.country"])))) 
    v["i34"] = 0.100000*np.tanh(((np.maximum(((data["totals.pageviews"])), ((data["trafficSource.isTrueDirect"])))) - (np.maximum(((data["trafficSource.isTrueDirect"])), ((data["totals.pageviews"])))))) 
    v["i35"] = 0.100000*np.tanh((((3.0) < (data["totals.hits"]))*1.)) 
    v["i36"] = 0.100000*np.tanh(((np.minimum(((((((((data["source.country"]) * 2.0)) * 2.0)) + (data["trafficSource.keyword"])))), ((data["totals.pageviews"])))) * 2.0)) 
    v["i37"] = 0.100000*np.tanh(((((((((-1.0) > ((((np.tanh((np.tanh((np.tanh(((6.0)))))))) > (-3.0))*1.)))*1.)) / 2.0)) > (data["trafficSource.source"]))*1.)) 
    v["i38"] = 0.100000*np.tanh(np.minimum(((((data["visitNumber"]) + (data["browser.os"])))), ((((((-3.0) + (data["totals.pageviews"]))) + (data["geoNetwork.region"])))))) 
    v["i39"] = 0.100000*np.tanh(((np.maximum(((((2.0) + (data["geoNetwork.continent"])))), ((0.0)))) * (((data["totals.hits"]) - (data["totals.pageviews"]))))) 
    v["i40"] = 0.100000*np.tanh(np.where(((data["trafficSource.isTrueDirect"]) + (((data["geoNetwork.metro"]) + (((data["trafficSource.isTrueDirect"]) + (data["visitNumber"]))))))<0, -2.0, data["geoNetwork.country"] )) 
    v["i41"] = 0.100000*np.tanh((((((((np.minimum(((((data["totals.bounces"]) * 2.0))), ((data["totals.newVisits"])))) * 2.0)) * 2.0)) + (data["geoNetwork.metro"]))/2.0)) 
    v["i42"] = 0.100000*np.tanh(np.where(((data["source.country"]) + (data["visitNumber"]))>0, data["totals.bounces"], -2.0 )) 
    v["i43"] = 0.100000*np.tanh((((data["channelGrouping"]) > (data["totals.pageviews"]))*1.)) 
    v["i44"] = 0.100000*np.tanh(((data["source.country"]) + (((((data["source.country"]) * 2.0)) + (np.where(data["trafficSource.isTrueDirect"]<0, ((data["source.country"]) * 2.0), ((data["device.operatingSystem"]) * 2.0) )))))) 
    v["i45"] = 0.100000*np.tanh(np.where(data["geoNetwork.country"]>0, ((((2.0)) > (data["trafficSource.source"]))*1.), 3.0 )) 
    v["i46"] = 0.100000*np.tanh(((((data["totals.pageviews"]) - ((5.0)))) * 2.0)) 
    v["i47"] = 0.100000*np.tanh((-1.0*((((((data["trafficSource.source"]) * (((data["month.unique.user.count"]) / 2.0)))) + (((data["source.country"]) * (data["month.unique.user.count"])))))))) 
    v["i48"] = 0.097362*np.tanh((((data["browser.os"]) + (((data["totals.pageviews"]) - (np.where(data["totals.hits"]>0, data["totals.hits"], data["device.operatingSystem"] )))))/2.0)) 
    v["i49"] = 0.100000*np.tanh(np.where(data["geoNetwork.continent"]<0, ((np.minimum(((data["totals.bounces"])), ((data["geoNetwork.continent"])))) * 2.0), data["totals.bounces"] )) 
    v["i50"] = 0.100000*np.tanh(((-3.0) + (((np.where(((data["geoNetwork.metro"]) * 2.0)<0, np.where(-2.0<0, data["totals.newVisits"], data["geoNetwork.metro"] ), data["geoNetwork.metro"] )) * 2.0)))) 
    v["i51"] = 0.100000*np.tanh(np.tanh((np.tanh((-1.0))))) 
    v["i52"] = 0.100000*np.tanh(((data["totals.bounces"]) + (((((((data["totals.pageviews"]) - (data["totals.hits"]))) * 2.0)) * (data["totals.bounces"]))))) 
    v["i53"] = 0.100000*np.tanh((((((np.minimum(((data["visitNumber"])), ((data["geoNetwork.networkDomain"])))) + (data["totals.bounces"]))/2.0)) * 2.0)) 
    v["i54"] = 0.099951*np.tanh((((2.0)) - (data["geoNetwork.region"]))) 
    v["i55"] = 0.099951*np.tanh(((data["totals.pageviews"]) + (np.minimum(((data["mean.pageviews.per.network.domain"])), ((np.where(data["geoNetwork.continent"]<0, (5.0), ((np.minimum(((data["geoNetwork.continent"])), ((data["geoNetwork.networkDomain"])))) * 2.0) ))))))) 
    v["i56"] = 0.100000*np.tanh(((data["source.country"]) + (((np.where(data["totals.pageviews"]<0, data["geoNetwork.city"], ((data["totals.pageviews"]) - (data["source.country"])) )) * 2.0)))) 
    v["i57"] = 0.100000*np.tanh((((-1.0*((((data["totals.pageviews"]) - (np.where(np.tanh((data["count.hits.per.network.domain"])) < -99998, data["totals.pageviews"], data["totals.hits"] ))))))) * (((data["totals.hits"]) * 2.0)))) 
    v["i58"] = 0.100000*np.tanh((((data["source.country"]) + (((np.maximum(((data["device.operatingSystem"])), ((((data["geoNetwork.region"]) + (data["source.country"])))))) + (((data["source.country"]) + (data["source.country"]))))))/2.0)) 
    v["i59"] = 0.100000*np.tanh(((((((data["channelGrouping"]) - (data["totals.pageviews"]))) + (3.0))) / 2.0)) 
    v["i60"] = 0.100000*np.tanh(((((((-2.0) < (((data["month.unique.user.count"]) - ((-1.0*((0.0)))))))*1.)) > (((((7.0)) < (data["geoNetwork.city"]))*1.)))*1.)) 
    v["i61"] = 0.100000*np.tanh(((((((np.where(data["totals.pageviews"]>0, ((data["totals.pageviews"]) - (data["channelGrouping"])), data["geoNetwork.country"] )) - (data["totals.hits"]))) * 2.0)) - (data["channelGrouping"]))) 
    v["i62"] = 0.100000*np.tanh(np.tanh((np.maximum((((-1.0*((data["geoNetwork.country"]))))), (((5.0)))))))
    return v.sum(axis=1)
cm = plt.cm.get_cmap('Greys')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(GP1(gp_train_df),
                  GP2(gp_train_df),
                  alpha=.5,
                  c=(gp_train_df['totals.transactionRevenue']),
                  cmap=cm,
                  s=30)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")
train_df['gp1'] = GP1(gp_train_df).values
train_df['gp2'] = GP2(gp_train_df).values
test_df['gp1'] = GP1(gp_test_df).values
test_df['gp2'] = GP2(gp_test_df).values
del gp_train_df
del gp_test_df
gc.collect()
train_df = train_df.sort_values('date')
X = train_df.drop(not_used_cols, axis=1)
y = train_df['totals.transactionRevenue']
X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns

lgb_params = {'num_leaves': 300,
             'min_data_in_leaf': 30, 
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 50,
             "boosting": "rf",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.8,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 1,
             "verbosity": -1
             }
             
lgb_params1 = {"objective" : "regression", "metric" : "rmse", "max_depth": 5, "min_child_samples": 100, "reg_alpha": 1, "reg_lambda": 1,
        "num_leaves" : 257, "learning_rate" : 0.01, "subsample" : 0.8, "colsample_bytree" : 0.8, "verbosity": -1}

run_lgb = True
print('LGB : ', run_lgb)
# modeling
#--------------------------------------------------------------------------
if run_lgb:
    import lightgbm as lgb
    def kfold_lgb_xgb():
        folds = KFold(n_splits=5, shuffle=True, random_state=7)
        
        oof_lgb = np.zeros(len(train_df))
        predictions_lgb = np.zeros(len(test_df))

        features_lgb = list(X.columns)
        feature_importance_df_lgb = pd.DataFrame()

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X)):
            trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
            val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
            
            print("LGB " + str(fold_) + "-" * 50)
            num_round = 20000
            clf = lgb.train(lgb_params1, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
            oof_lgb[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)
            
            fold_importance_df_lgb = pd.DataFrame()
            fold_importance_df_lgb["feature"] = features_lgb
            fold_importance_df_lgb["importance"] = clf.feature_importance()
            fold_importance_df_lgb["fold"] = fold_ + 1
            feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
            predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        
        #lgb.plot_importance(clf, max_num_features=30)    
        cols = feature_importance_df_lgb[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
        best_features_lgb = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]
        plt.figure(figsize=(14,10))
        sns.barplot(x="importance", y="feature", data=best_features_lgb.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')
        x = []
        for i in oof_lgb:
            if i < 0:
                x.append(0.0)
            else:
                x.append(i)
        cv_lgb = mean_squared_error(x, y)**0.5
        cv_lgb = str(cv_lgb)
        cv_lgb = cv_lgb[:10]
        
        pd.DataFrame({'preds': x}).to_csv('lgb_oof_' + cv_lgb + '.csv', index = False)
        
        print("CV_LGB : ", cv_lgb)
        return cv_lgb, predictions_lgb
        
    cv_lgb, lgb_ans = kfold_lgb_xgb()
    x = []
    for i in lgb_ans:
        if i < 0:
            x.append(0.0)
        else:
            x.append(i)
    np.save('lgb_ans.npy', x)
    submission = test_df[['fullVisitorId']].copy()
    submission.loc[:, 'PredictedLogRevenue'] = x
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
    grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    grouped_test.to_csv('lgb_' + cv_lgb + '.csv',index=False)