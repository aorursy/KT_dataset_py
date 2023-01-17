%matplotlib inline

import zipfile

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost

import zipfile



from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
Dataset = "train"

Dataset1 = "test"

Dataset2 = "sample_submission"



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/nyc-taxi-trip-duration/"+Dataset+".zip","r") as z:

    z.extractall("./")



with zipfile.ZipFile("../input/nyc-taxi-trip-duration/"+Dataset1+".zip","r") as z:

    z.extractall("./")



with zipfile.ZipFile("../input/nyc-taxi-trip-duration/"+Dataset2+".zip","r") as z:

    z.extractall("./")
df = pd.read_csv('train.csv')

df.passenger_count = df.passenger_count.astype(np.uint8)

df.vendor_id = df.vendor_id.astype(np.uint8)

df.trip_duration = df.trip_duration.astype(np.uint32)

for c in [c for c in df.columns if c.endswith('tude')]:

    df.loc[:,c] = df[c].astype(np.float32)

print(df.memory_usage().sum()/2**20)

df.pickup_datetime=pd.to_datetime(df.pickup_datetime)

df.dropoff_datetime=pd.to_datetime(df.dropoff_datetime)

df['pu_hour'] = df.pickup_datetime.dt.hour

df['yday'] = df.pickup_datetime.dt.dayofyear

df['wday'] = df.pickup_datetime.dt.dayofweek

df['month'] = df.pickup_datetime.dt.month
sns.set_style('white')

sns.set_context("paper",font_scale=2)

corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,

           square=True, linewidths=0.5, cbar_kws={"shrink":0.5})
fig, ax = plt.subplots(ncols=1, nrows=1)

sns.distplot(df['trip_duration']/3600,ax=ax,bins=100,kde=False,hist_kws={'log':True})
fig, ax = plt.subplots(ncols=1, nrows=1)

ax.set_xlim(0,30)

sns.distplot(df['trip_duration']/3600,ax=ax,bins=1000,kde=False,hist_kws={'log':True})
def haversine(lon1, lat1, lon2, lat2):



    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    miles = km *  0.621371

    return miles
df['distance'] = haversine(df.pickup_longitude, df.pickup_latitude,

                                           df.dropoff_longitude, df.dropoff_latitude)
sns.set_style('white')

sns.set_context("paper",font_scale=2)

corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,

           square=True, linewidths=0.5, cbar_kws={"shrink":0.5})
corr
mask = ((df.trip_duration > 60) & (df.distance < 0.05))

df = df[~mask]

mask = (df.trip_duration < 60) 

df = df[~mask]

mask =  df.trip_duration > 79200

df = df[~mask]

mask = df.distance/(df.trip_duration/3600) > 60

df = df[~mask]

df.trip_duration = df.trip_duration.astype(np.uint16)

df = df[df.passenger_count > 0]
m = df.groupby(['wday','vendor_id'])[['trip_duration']].apply(np.median)

m.name = 'trip_duration_median'

df = df.join(m, on=['wday','vendor_id'])
sns.lmplot(y='trip_duration_median', x='wday',data=df, fit_reg=False, hue='vendor_id')
m = df.groupby(['pu_hour','vendor_id'])[['trip_duration']].apply(np.median)

m.name ='trip_duration_median_hour'

df = df.join(m, on=['pu_hour','vendor_id'])
sns.lmplot(y='trip_duration_median_hour', x='pu_hour',data=df, fit_reg=False, hue='vendor_id')
jfk_lon = -73.778889

jfk_lat = 40.639722

lga_lon = -73.872611

lga_lat = 40.77725
df['jfk_pickup_dist'] = df.apply(lambda row: haversine(jfk_lon, jfk_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)

df['lga_pickup_dist'] = df.apply(lambda row: haversine(lga_lon, lga_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)

df['jfk_dropoff_dist'] = df.apply(lambda row: haversine(jfk_lon, jfk_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

df['lga_dropoff_dist'] = df.apply(lambda row: haversine(lga_lon, lga_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)
fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True)

ax[0,0].set_xlim(0,50)



sns.distplot(df['jfk_pickup_dist'],ax=ax[0,0],bins=100,kde=False,hist_kws={'log':True})

sns.distplot(df['jfk_dropoff_dist'],ax=ax[0,1],bins=100,kde=False,hist_kws={'log':True})

sns.distplot(df['lga_pickup_dist'],ax=ax[1,0],bins=100,kde=False,hist_kws={'log':True})

sns.distplot(df['lga_dropoff_dist'],ax=ax[1,1],bins=100,kde=False,hist_kws={'log':True})
df['jfk'] = ((df['jfk_pickup_dist'] < 2) | (df['jfk_dropoff_dist'] < 2))

df['lga'] = ((df['lga_pickup_dist'] < 2) | (df['lga_dropoff_dist'] < 2))

df = df.drop(['jfk_pickup_dist','lga_pickup_dist','jfk_dropoff_dist','lga_dropoff_dist'],axis=1)

df.head()
df['workday'] = ((df['pu_hour'] > 8) & (df['pu_hour'] < 18))

df.head()
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

ax.scatter(df['pickup_longitude'],df['pickup_latitude'], s=0.01, alpha=1)
def rmsle(evaluator,X,real):

    sum = 0.0

    predicted = evaluator.predict(X)

    print("Number predicted less than 0: {}".format(np.where(predicted < 0)[0].shape))



    predicted[predicted < 0] = 0

    for x in range(len(predicted)):

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p-r)**2

    return (sum/len(predicted))**0.5
tdf = pd.read_csv('test.csv')

tdf.pickup_datetime=pd.to_datetime(tdf.pickup_datetime)

#tdf.dropoff_datetime=pd.to_datetime(tdf.dropoff_datetime)

tdf['pu_hour'] = tdf.pickup_datetime.dt.hour

tdf['yday'] = tdf.pickup_datetime.dt.dayofyear

tdf['wday'] = tdf.pickup_datetime.dt.dayofweek

tdf['month'] = tdf.pickup_datetime.dt.month

tdf['distance'] = haversine(tdf.pickup_longitude, tdf.pickup_latitude,

                                           tdf.dropoff_longitude, tdf.dropoff_latitude)





tdf['jfk_pickup_dist'] = tdf.apply(lambda row: haversine(jfk_lon, jfk_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)

tdf['lga_pickup_dist'] = tdf.apply(lambda row: haversine(lga_lon, lga_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)

tdf['jfk_dropoff_dist'] = tdf.apply(lambda row: haversine(jfk_lon, jfk_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

tdf['lga_dropoff_dist'] = tdf.apply(lambda row: haversine(lga_lon, lga_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

tdf['jfk'] = ((tdf['jfk_pickup_dist'] < 2) | (tdf['jfk_dropoff_dist'] < 2))

tdf['lga'] = ((tdf['lga_pickup_dist'] < 2) | (tdf['lga_dropoff_dist'] < 2))

tdf = tdf.drop(['jfk_pickup_dist','lga_pickup_dist','jfk_dropoff_dist','lga_dropoff_dist'],axis=1)

tdf['workday'] = ((tdf['pu_hour'] > 8) & (tdf['pu_hour'] < 18))
tdf.head()
df.to_csv('train_data.csv',index=False)

tdf.to_csv('test_data.csv',index=False)
df = pd.read_csv('train_data.csv')

tdf = pd.read_csv('test_data.csv')
features = df[['vendor_id','passenger_count','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pu_hour','wday','month','workday','jfk','lga']]

target = df['trip_duration']
tfeatures = tdf[['vendor_id','passenger_count','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pu_hour','wday','month','workday','jfk','lga']]
from sklearn import linear_model



reg = linear_model.LinearRegression()

cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)

print(cross_val_score(reg, features, np.ravel(target), cv=cv, scoring=rmsle))

reg.fit(features, target)
np.mean([0.43999617,  0.44022755 , 0.43897449,0.44073678])
tfeatures.shape
pred = reg.predict(tfeatures)

print(np.where(pred < 0)[0].shape)

pred[pred < 0]=0
tdf['trip_duration']=pred.astype(int)

out = tdf[['id','trip_duration']]
out['trip_duration'].isnull().values.any()
out.to_csv('pred_linear.csv',index=False)

#RMSLE=0.535
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=10)

cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)

print(cross_val_score(neigh, features, np.ravel(target), cv=cv,scoring=rmsle))

neigh.fit(features,target)
np.mean([0.42010954, 0.41940803, 0.41947931, 0.41968818])
pred = neigh.predict(tfeatures)

print(np.where(pred < 0)[0].shape)
tdf['trip_duration']=pred.astype(int)

out = tdf[['id','trip_duration']]

out.to_csv('pred_knn.csv',index=False)

#RMSLE=0.505
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)

print(cross_val_score(rf, features, np.ravel(target), cv=cv,scoring=rmsle))

rf = rf.fit(features,np.ravel(target))
np.mean([0.35085909, 0.35181223, 0.34976928, 0.35057147])
pred = rf.predict(tfeatures)

print(np.where(pred < 0)[0].shape)
tdf['trip_duration']=pred.astype(int)

out = tdf[['id','trip_duration']]

out.to_csv('pred_rf.csv',index=False)

#RMSLE=0.473
reg = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)



cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)

print(cross_val_score(reg, features, np.ravel(target), cv=cv,scoring=rmsle))

reg.fit(features,target)
pred = reg.predict(tfeatures)

print(np.where(pred < 0)[0].shape)
pred[pred < 0] = 0

tdf['trip_duration']=pred.astype(int)

out = tdf[['id','trip_duration']]

out['trip_duration'].isnull().values.any()

out.to_csv('pred_xgboost.csv',index=False)

#RMSLE=0.463
from xgboost import plot_tree

plot_tree(reg)
import pickle

pickle.dump(reg, open('xgb_model.sav','wb'),protocol=2)
# DISCLAIMER: THIS CODE IS A RESULT OF FOLLOWING THIS TUTORIAL: https://www.youtube.com/watch?v=suRd3UzdBeo