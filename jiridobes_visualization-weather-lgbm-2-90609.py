import os
import zipfile
import pandas as pd
import numpy as np
#import time
#import geopy
#from geopy import distance

import random

from sklearn.metrics import mean_squared_error, r2_score

import math
from math import sqrt

import pickle

import lightgbm as lgb

import matplotlib.pyplot as plt
import folium
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

pd.set_option('mode.chained_assignment','raise')
# os.chdir("C:/Users/...")
    
def read_files(chunks = 1100, chunk_no = None):
    random.seed(42)

    nrows = int( 55423857 / chunks )

    def skiprows(x): # This can do true random selection from the dataset. Useful for development if the data set was sorted.
        if x == 0:
            return False
        return chunk_no != random.randrange(chunks)

    df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = nrows, parse_dates=["pickup_datetime"])
        
#    with zipfile.ZipFile('all.zip') as zipf: 
#         with zipf.open('train.csv') as myZip:
#             if chunk_no == None:
#                 df = pd.read_csv(myZip,nrows=nrows,parse_dates=["pickup_datetime"])
#             else:
#                 df = pd.read_csv(myZip,skiprows=skiprows,parse_dates=["pickup_datetime"])

    random.seed(42)
    df = df.sample(frac=1).reset_index(drop=True) #Random reshuffle
    
    df_test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv', nrows = nrows, parse_dates=["pickup_datetime"])
                
#    with zipfile.ZipFile('all.zip') as zipf: 
#        with zipf.open('test.csv') as myZip:
#            df_test = pd.read_csv(myZip,parse_dates=["pickup_datetime"])
                
    return df, df_test
chunks = 500 # Run chunks = 1 for real, accurate results.  
df,df_test = read_files(chunks)
print("Data loaded...")
df.head()
df.describe()
df_test.head()
def calculate_distance(data):
    
# https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    for df in data:
        R=6371.0
        pi=math.pi

        sa = math.sin(-30.*pi/360.)
        ca = math.cos(-30.*pi/360.)

        # Wikipedia NYC
        # 40.7127, -74.0059

        df_tmp = df.eval("""
        lat1 = @pi / 180. * pickup_latitude
        lon1 = @pi / 180. * pickup_longitude
        lat2 = @pi / 180. * dropoff_latitude
        lon2 = @pi / 180. * dropoff_longitude

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * arctan2(sqrt(a), sqrt(1 - a))

        distance = @R * c
        
        ty1 = (pickup_latitude - 40.7127) * 30.77915051090274   #y
        tx1 = (pickup_longitude + 74.0059) * 111.60876733592401 #x
        ty2 = (dropoff_latitude - 40.7127) * 30.77915051090274
        tx2 = (dropoff_longitude + 74.0059) * 111.60876733592401 
        
        x1 = tx1 * @ca - ty1 * @sa
        y1 = ty1 * @ca + tx1 * @sa
        x2 = tx2 * @ca - ty2 * @sa
        y2 = ty2 * @ca + tx2 * @sa
        
        """,inplace=False)
        df['dist'] = df_tmp['distance']
        df['dist2'] = df_tmp['distance'] * df_tmp['distance'] 
        df[['x1','y1','x2','y2']] = df_tmp[['x1','y1','x2','y2']]
    return data
# geopy.distance.distance((-74.1, 40.),(-74., 40.)).km # 11.160876733592401
# geopy.distance.distance((-74., 40.1),(-74., 40.)).km # 3.077915051090274

def is_airport(data):
    long_km = .1 / 11.160876733592401 # one over...
    lat_km = .1 / 3.077915051090274

    for df in data:
        df["is_jfk_pickup"] = False
        df.loc[df.eval(
            '(-73.778889 - 2*@long_km < pickup_longitude < -73.778889 + 2*@long_km)\
            &\
            (40.639722 - 1*@lat_km < pickup_latitude < 40.639722 + .5*@lat_km)')
               ,"is_jfk_pickup"] = True

        df["is_ewr_pickup"] = False
        df.loc[df.eval(
            '(-74.168611 - 2*@long_km < pickup_longitude < -74.168611 + 2*@long_km)\
            &\
            (40.6925 - 1*@lat_km < pickup_latitude < 40.6925 + 1*@lat_km)')
               ,"is_ewr_pickup"] = True

        df["is_lga_pickup"] = False
        df.loc[df.eval(
            '(-73.872611 - 1.4*@long_km < pickup_longitude < -73.872611 + 1.5*@long_km)\
            &\
            (40.77725 - .3*@lat_km < pickup_latitude < 40.77725 + .3*@lat_km)')
               ,"is_lga_pickup"] = True

        
        df["is_jfk_dropoff"] = False
        df.loc[df.eval(
            '(-73.778889 - 2*@long_km < dropoff_longitude < -73.778889 + 2*@long_km)\
            &\
            (40.639722 - 1*@lat_km < dropoff_latitude < 40.639722 + .5*@lat_km)')
               ,"is_jfk_dropoff"] = True

        df["is_ewr_dropoff"] = False
        df.loc[df.eval(
            '(-74.168611 - 2*@long_km < dropoff_longitude < -74.168611 + 2*@long_km)\
            &\
            (40.6925 - 1*@lat_km < dropoff_latitude < 40.6925 + 1*@lat_km)')
               ,"is_ewr_dropoff"] = True

        df["is_lga_dropoff"] = False
        df.loc[df.eval(
            '(-73.872611 - 1.4*@long_km < dropoff_longitude < -73.872611 + 1.5*@long_km)\
            &\
            (40.77725 - .3*@lat_km < dropoff_latitude < 40.77725 + .3*@lat_km)')
               ,"is_lga_dropoff"] = True
    return data


df,df_test = calculate_distance([df,df_test])
df,df_test = is_airport([df,df_test])
print("Airports defined...")
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

def time_related_vars(data):
    for df in data:
        # df['bday'] = df['pickup_datetime'].apply(is_business_day) # useful feature, but awefully slow...
        #df['bday'] = bool(len(pd.bdate_range(df['pickup_datetime'].value(), df['pickup_datetime'].value())))
        df['weekday'] = df['pickup_datetime'].dt.weekday # The day of the week with Monday=0, Sunday=6
        df['day'] = df['pickup_datetime'].dt.day
        df['hour'] = df['pickup_datetime'].dt.hour
        df['month'] = df['pickup_datetime'].dt.month
        df['year'] = df['pickup_datetime'].dt.year
        df['daytime'] = False
        df.loc[(df.hour >= 8)&(df.hour < 20),'daytime'] = True 
        #df['businesstime'] = False
        #df.loc[df.daytime&df.bday,'businesstime'] = True 
        #df['time_stamp'] = df['year'].astype('str') + '-' + df['month'].astype('str') + '-' + df['day'].astype('str') # if running on small subset
        df['time_stamp'] = df['year'].astype('str') + '-' + df['month'].astype('str') + '-' + df['day'].astype('str') + ':' + df['hour'].astype('str')
    return data
df,df_test = time_related_vars((df,df_test))
# Free inspiration from
# https://www.kaggle.com/nicapotato/taxi-rides-time-analysis-and-oof-lgbm
grp = df.groupby('time_stamp',as_index=False)['fare_amount'].agg({'tsum':'sum','tmean':'mean','tstd':'std','tskew':'skew','tcount':'count'}).reset_index()
df = df.merge(grp, how='left', on=['time_stamp'])
df_test = df_test.merge(grp, how='left', on=['time_stamp'])
#df[df.isnull().any(axis=1)]
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

# geopy.distance.distance((-74.1, 40.),(-74., 40.)).km # 11.160876733592401
# geopy.distance.distance((-74., 40.1),(-74., 40.)).km # 3.077915051090274

def space_related_vars(data): # not used for winning solution.
    long_km = 111.60876733592401
    lat_km = 30.77915051090274
    mult_km = 0.5 

    for df in data:
        df['space_stamp'] = \
        (lat_km*df['pickup_latitude']/mult_km).astype("int").astype("str") + ':' + \
        (long_km*df['pickup_longitude']/mult_km).astype("int").astype("str") + '-' +\
        (lat_km*df['dropoff_latitude']/mult_km).astype("int").astype("str") + ':' + \
        (long_km*df['dropoff_longitude']/mult_km).astype("int").astype("str") 
    return data

df,df_test = space_related_vars((df,df_test))
df.head()
# Free inspiration from
# https://www.kaggle.com/nicapotato/taxi-rides-time-analysis-and-oof-lgbm
grp = df.groupby('space_stamp',as_index=False)['fare_amount'].agg({'ssum':'sum','smean':'mean','sstd':'std','sskew':'skew','scount':'count'}).reset_index()
#grp = df.groupby('space_stamp',as_index=False)['fare_amount'].agg({'ssum':'sum','scount':'count'}).reset_index()
df = df.merge(grp, how='left', on=['space_stamp'])
df_test = df_test.merge(grp, how='left', on=['space_stamp'])
df.head()
def basic_fare():
    df['min_fare'] = 2.5 + 0.5 * 5 * df.dist / 1.6
    df_test['min_fare'] = 2.5 + 0.5 * 5 * df_test.dist / 1.6
    df['min_multiple'] = df['fare_amount'] / df['min_fare']
basic_fare()
wdf = pd.read_csv('../input/nyc-weather-20092015/1428442.csv',usecols=['DATE','HOURLYVISIBILITY','HOURLYDRYBULBTEMPC','HOURLYWindSpeed',
    'HOURLYPrecip','DAILYPrecip','DAILYSnowfall','DAILYSnowDepth'],dtype={'HOURLYVISIBILITY':str,'HOURLYDRYBULBTEMPC':str,
    'HOURLYPrecip':str,'DAILYPrecip':str,'DAILYSnowfall':str,'DAILYSnowDepth':str})
wdf.head()
wdf['HOURLYVISIBILITY'] = wdf['HOURLYVISIBILITY'].str.replace('V','') # see the dataset documentation. T = trace etc.
wdf['HOURLYDRYBULBTEMPC'] = wdf['HOURLYDRYBULBTEMPC'].str.replace('s','')
wdf['HOURLYPrecip'] = wdf['HOURLYPrecip'].str.replace('s','')
wdf['HOURLYPrecip'] = wdf['HOURLYPrecip'].str.replace('T','0')
wdf['DAILYPrecip'] = wdf['DAILYPrecip'].str.replace('T','0')
wdf['DAILYSnowfall'] = wdf['DAILYSnowfall'].str.replace('T','0')
wdf['DAILYSnowDepth'] = wdf['DAILYSnowDepth'].str.replace('T','0')
wdf[['HOURLYVISIBILITY','HOURLYDRYBULBTEMPC','HOURLYWindSpeed','HOURLYPrecip','DAILYPrecip','DAILYSnowfall','DAILYSnowDepth']]=wdf[['HOURLYVISIBILITY','HOURLYDRYBULBTEMPC','HOURLYWindSpeed','HOURLYPrecip','DAILYPrecip','DAILYSnowfall','DAILYSnowDepth']].astype(np.float)
wdf['DATE']=pd.to_datetime(wdf['DATE'])
wdf=wdf.set_index('DATE')
wdf=wdf.resample('1H').nearest()
wdf=wdf.fillna(method='ffill')
wdf=wdf.fillna(method='bfill')
wdf.head()
wdf=wdf.reset_index()
wdf.head()
wdf['time_stamp'] = wdf['DATE'].dt.year.astype('str') + '-' + wdf['DATE'].dt.month.astype('str') + '-' + wdf['DATE'].dt.day.astype('str') + ':' + wdf['DATE'].dt.hour.astype('str')
wdf.head()
df = df.merge(wdf, how='left', on=['time_stamp'])
df_test = df_test.merge(wdf, how='left', on=['time_stamp'])
df.head()
print( "NANs: {}, {}".format(len(df[df.isnull().any(axis=1)]),len(df_test[df_test.isnull().any(axis=1)])))
df[df.isnull().any(axis=1)].head()

# df.to_pickle('df.pkl')
# df_test.to_pickle('df_test.pkl')
# df = pd.read_pickle('df.pkl')
# df_test = pd.read_pickle('df_test.pkl')
df.dropna(inplace=True)

df.drop(df.index[(df.pickup_longitude < -75) | 
           (df.pickup_longitude > -72) | 
           (df.pickup_latitude < 40) | 
           (df.pickup_latitude > 42)],inplace=True)
df.drop(df.index[(df.dropoff_longitude < -75) | 
           (df.dropoff_longitude > -72) | 
           (df.dropoff_latitude < 40) | 
           (df.dropoff_latitude > 42)],inplace=True)
df.nlargest(10,'fare_amount')
df.nsmallest(20,'fare_amount')
df[['fare_amount', 'dist']].plot.scatter(x='dist',y='fare_amount')
# drop all x more expensive than it should be. No problem with airports, it easily fits:
# Initial fare $2.5 + $0.5 per 1/5 mile
max_fare_multiple = 20
min_fare_multiple = 0.05
drop_too_expensive = df.index[(df.fare_amount > ( max_fare_multiple * ( 2.5 + 0.5 * 5 * df.dist / 1.6 )))] 
print("n_dropped={}, {}%".format(len(drop_too_expensive),len(drop_too_expensive)/len(df)))
df.drop(drop_too_expensive,inplace=True)
# And of the minimum
drop_too_cheap = df.index[df.fare_amount < ( min_fare_multiple * ( 2.5 + 0.5 * 5 * df.dist / 1.6 ))] 
print("n_dropped={}, {}%".format(len(drop_too_cheap),len(drop_too_cheap)/len(df)))
df.drop(drop_too_cheap,inplace=True)
df[['fare_amount', 'dist']].plot.scatter(x='dist',y='fare_amount')
#from folium.plugins import FastMarkerCluster

def plot_map(df=df, maxpoints=200,showlines=False):
    m = folium.Map(
        location=[df["pickup_latitude"].median(), df["pickup_longitude"].median()],
        zoom_start=12)

    for index, row in enumerate(list(zip(df["pickup_latitude"].values, df["pickup_longitude"].values))):
        folium.CircleMarker(location=row, radius=2, weight=1, color='green').add_to(m)
        if index == maxpoints:
            break

    for index, row in enumerate(list(zip(df["dropoff_latitude"].values, df["dropoff_longitude"].values))):
        folium.CircleMarker(location=row, radius=2, weight=1, color='red').add_to(m)
        if index == maxpoints:
            break

            
    if showlines:
        for index, row in enumerate(list(
            zip(zip(df["dropoff_latitude"].values, df["dropoff_longitude"].values),
                zip(df["pickup_latitude"].values, df["pickup_longitude"].values)))):
            #folium.CircleMarker(location=row, radius=2, weight=1, color='red').add_to(m)
            #print(index,row)
            folium.PolyLine(row, color="blue", weight=1, opacity=0.5).add_to(m)
            if index == maxpoints:
                break
            
    return m
                
plot_map(df_test,80,True)
plt.figure();
df.dist.plot.hist(bins=50, range=(0,100))
plt.figure();
df_test.dist.plot.hist(bins=50,range=(0,100))
plt.figure();
df.dist.plot.hist(bins=50, range=(0,25))
plt.figure();
df_test.dist.plot.hist(bins=50,range=(0,25))
df.pickup_datetime.describe()
df_test.pickup_datetime.describe()
df[['fare_amount','weekday']].boxplot(by='weekday',showfliers=False)
# df[['fare_amount','bday']].boxplot(by='bday',showfliers=False)
# df[['dist','businesstime']].boxplot(by='businesstime',showfliers=False)
# df[['fare_amount','businesstime']].boxplot(by='businesstime',showfliers=False)
df[['dist','hour']].boxplot(by='hour',showfliers=False)
df[['fare_amount','hour']].boxplot(by='hour',showfliers=False)
df[['dist','month']].boxplot(by='month',showfliers=False)
df[['fare_amount','month']].boxplot(by='month',showfliers=False)
df.head()
df.describe()
# Solver cannot run with time, needs to have the float.
df['float_pickup_datetime'] = df['pickup_datetime'].values.astype('float')
df_test['float_pickup_datetime'] = df_test['pickup_datetime'].values.astype('float')
# Random split to train and test
dftr = df.sample( frac = 0.8, random_state=42)
dfval = df.drop( dftr.index)
# This way interativelly trial and error select a good set of features.
ll = ['passenger_count','dist','bday','weekday','hour','daytime','businesstime','month','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','lat_diff','long_diff']
ll = ['dist'] # 4.971843085485664
ll = ['passenger_count','dist'] # 4.978781923333827 -> not important
ll = ['dist','bday'] # 4.938947744089244 -> OK
ll = ['dist','bday','weekday'] # 4.910031675409393 -> OK
ll = ['dist','bday','weekday','hour'] # 4.837025815294643 -> OK
ll = ['dist','bday','weekday','hour','daytime'] # 4.759085591019987 -> OK
ll = ['dist','bday','weekday','hour','businesstime'] # 4.804090429646956 -> Not OK
ll = ['dist','bday','weekday','hour','daytime','businesstime'] # 4.79580778655997 -> OK
ll = ['dist','month'] # 4.955495881088339 -> OK
ll = ['dist','bday','weekday','hour','daytime','month'] # 4.706722506235075 -> OK

ll = ['dist','bday','weekday','hour','daytime','month','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'] # 4.284077318471638 -> OK

ll = ['dist','bday','weekday','hour','daytime','month','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','lat_diff','long_diff'] # 4.315628577392338 -> Not OK


ll = ['dist','bday','weekday','hour','daytime','month','lat_avg','long_avg','lat_diff','long_diff']

ll = ['dist','bday','weekday','hour','daytime','month','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'] # 4.284077318471638 -> OK

ll = ['dist','dist2','weekday','hour','daytime','year','month','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
'is_jfk_pickup', 'is_ewr_pickup', 'is_lga_pickup', 'is_jfk_dropoff', 'is_ewr_dropoff', 'is_lga_dropoff']

ll = ['dist','dist2','float_pickup_datetime', 'weekday','hour','daytime','year','month','pickup_longitude',
      'pickup_latitude','dropoff_longitude','dropoff_latitude','is_jfk_pickup', 'is_ewr_pickup',
      'is_lga_pickup', 'is_jfk_dropoff', 'is_ewr_dropoff', 'is_lga_dropoff',
      'sum','mean','std','skew','count']

ll = ['dist','dist2','float_pickup_datetime', 'weekday','hour','daytime','year','month','pickup_longitude',
      'pickup_latitude','dropoff_longitude','dropoff_latitude','is_jfk_pickup', 'is_ewr_pickup',
      'is_lga_pickup', 'is_jfk_dropoff', 'is_ewr_dropoff', 'is_lga_dropoff',
      'sum','mean','std','skew','count','x1','y1','x2','y2']

ll = ['dist','dist2','float_pickup_datetime', 'weekday','hour','daytime','year','month','pickup_longitude',
      'pickup_latitude','dropoff_longitude','dropoff_latitude','is_jfk_pickup', 'is_ewr_pickup',
      'is_lga_pickup', 'is_jfk_dropoff', 'is_ewr_dropoff', 'is_lga_dropoff',
      'sum','mean','std','skew','count','x1','y1','x2','y2',
      'HOURLYVISIBILITY','HOURLYDRYBULBTEMPC','HOURLYWindSpeed',
      'HOURLYPrecip','DAILYPrecip','DAILYSnowfall','DAILYSnowDepth']

ll = ['dist','dist2','float_pickup_datetime', 'weekday','hour','daytime','year','month','pickup_longitude',
      'pickup_latitude','dropoff_longitude','dropoff_latitude','is_jfk_pickup', 'is_ewr_pickup',
      'is_lga_pickup', 'is_jfk_dropoff', 'is_ewr_dropoff', 'is_lga_dropoff',
      'tsum','tmean','tstd','tskew','tcount','ssum','smean','sstd','sskew','scount','x1','y1','x2','y2',
      'HOURLYVISIBILITY','HOURLYDRYBULBTEMPC','HOURLYWindSpeed',
      'HOURLYPrecip','DAILYPrecip','DAILYSnowfall','DAILYSnowDepth']

# Winning coctail:
ll = ['dist','dist2','float_pickup_datetime', 'weekday','hour','daytime','year','month','pickup_longitude',
      'pickup_latitude','dropoff_longitude','dropoff_latitude','is_jfk_pickup', 'is_ewr_pickup',
      'is_lga_pickup', 'is_jfk_dropoff', 'is_ewr_dropoff', 'is_lga_dropoff',
      'tsum','tmean','tstd','tskew','tcount','x1','y1','x2','y2',
      'HOURLYVISIBILITY','HOURLYDRYBULBTEMPC','HOURLYWindSpeed',
      'HOURLYPrecip','DAILYPrecip','DAILYSnowfall','DAILYSnowDepth']
ntr = len(dftr['dist'])
nval = len(dfval['dist'])
nte = len(df_test['dist'])
#ntotal = len(df['dist'])

Xtr = pd.DataFrame(dftr[ll]).values.reshape(ntr, len(ll))
Xval = pd.DataFrame(dfval[ll]).values.reshape(nval, len(ll))
Xte = pd.DataFrame(df_test[ll]).values.reshape(nte, len(ll))
#Xtotal = pd.DataFrame(df[ll]).values.reshape(ntotal, len(ll))

# We will actually learn min_multiple, as it is at the first approximaton constant 
Ytr = dftr['min_multiple'].values
Yval = dfval['min_multiple'].values
# https://lightgbm.readthedocs.io/en/latest/Python-API.html
# Good parameters are data size dependent. Should be found e.g. by grid search
max_bin = 2*255
n_estimators=10000
gbm = lgb.LGBMRegressor(n_estimators=n_estimators, silent=False, max_bin=max_bin)
print("Fitting...")
parameters = gbm.fit( Xtr, Ytr )
print("Fitted...")
# pickle.dump(gbm, open('model.pkl', 'wb'))
# gbm = pickle.load(open('model.pkl', 'rb'))
Ytest = gbm.predict(Xte)
df_test['fare_amount'] = pd.Series(Ytest) * df_test['min_fare']
df_test[['key','fare_amount']].to_csv('submission.csv',index=False)
print("Submission file created...")
# Print some statistics
print("FARE MULTIPLE, LGBM")
print(parameters)
print(ll)
print("df size: {}".format(len(df)))
print("chunks: {}".format(chunks))
print("max bin: {}".format(max_bin))
print("min,max_fare_multiple {} {}".format(min_fare_multiple,max_fare_multiple))
print("n_estimators={}".format(n_estimators))
Ytr_hat = gbm.predict(Xtr)
Yval_hat = gbm.predict(Xval)
# Ytotal = gbm.predict(Xtotal)
# df['fare_predicted'] = pd.Series(Ytotal) * df['min_fare']
dftr['fare_predicted'] = pd.Series(Ytr) * dftr['min_fare']
dfval['fare_predicted'] = pd.Series(Yval) * dfval['min_fare']


#df['fare_err'] = (df['fare_amount'] - df['fare_predicted']).abs()
dftr['fare_err'] = (dftr['fare_amount'] - dftr['fare_predicted']).abs()
dfval['fare_err'] = (dfval['fare_amount'] - dfval['fare_predicted']).abs()
rms_tr = (dftr['fare_err']**2).mean()**.5
print( "Accuracy training: ", rms_tr)
rms_val = (dfval['fare_err']**2).mean()**.5
print( "Accuracy testing: ", rms_val)
err100 = dfval.nlargest(100,'fare_err')
err100.to_csv('100errval.csv')
err100.head()
plot_map(err100,100,True)
plt.figure();
dftr.fare_err.plot.hist(bins=50, range=(0,25))
plt.figure();
dfval.fare_err.plot.hist(bins=50, range=(0,25))
dftr[['fare_err', 'dist']].plot.scatter(x='dist',y='fare_err')
dfval[['fare_err', 'dist']].plot.scatter(x='dist',y='fare_err')
print("DONE.")

