import numpy as np
import pandas as pd
import dask.dataframe as dd
import pandas_profiling
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly_express as px
import time
import random 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from vecstack import stacking
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance
import lime
import lime.lime_tabular
import shap
import geopandas as gpd
from shapely.geometry import Point,Polygon
import descartes
#Read data through Pandas and compute time taken to read
df_taxi = pd.read_csv('../input/nyc-taxi-trip-duration/train.zip',parse_dates=['pickup_datetime','dropoff_datetime'],infer_datetime_format=True)
#Getting the head of the dataset
df_taxi.head(5)
#Shape of the dataset
df_taxi.shape
#Data Type of features for dataset
df_taxi.dtypes
#Info of the dataset
df_taxi.info()
#Checking for null values in dataset
df_taxi.isnull().sum()
#Checking duplicate value in vendor_id
df_taxi[df_taxi.duplicated(['id'], keep=False)]
#Checking Date and Time range
print('Datetime range: {} to {}'.format(df_taxi.pickup_datetime.min(),df_taxi.dropoff_datetime.max()))
#Checking no. of vendors
df_taxi['vendor_id'].value_counts()
#Checking Passenger count
print('Passenger Count: {} to {}'.format(df_taxi.passenger_count.min(),df_taxi.passenger_count.max()))
#The distribution of Pickup and Drop Off day of the week
print(df_taxi['pickup_datetime'].nunique())
print(df_taxi['dropoff_datetime'].nunique())
#Performing Pandas profiling to understand quick overview of columns
report = pandas_profiling.ProfileReport(df_taxi)
#coverting profile report as html file
report.to_file('taxi_train.html')

from IPython.display import display,HTML,IFrame
display(HTML(open('taxi_train.html').read()))  
#Summary statistics for the dataset
df_taxi.describe()
#Passenger Count
sns.distplot(df_taxi['passenger_count'],kde=False)
plt.title('Distribution of Passenger Count')
plt.show()
#Creating pickup and dropoff day
df_taxi['pickup_day']=df_taxi['pickup_datetime'].dt.day_name()
df_taxi['dropoff_day']=df_taxi['dropoff_datetime'].dt.day_name()
#Creating pickup and dropoff month
df_taxi['pickup_month']=df_taxi['pickup_datetime'].dt.month
df_taxi['dropoff_month']=df_taxi['dropoff_datetime'].dt.month
#Creating pickup and dropoff hour
df_taxi['pickup_hour']=df_taxi['pickup_datetime'].dt.hour
df_taxi['dropoff_hour']=df_taxi['dropoff_datetime'].dt.hour
df_taxi.head(2)
#Plotting monthly Pickup and Dropoff trip distribution
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,4))
sns.countplot(x='pickup_month',data=df_taxi,ax=ax[0])
ax[0].set_title('The distribution of number of pickups each month')
sns.countplot(x='dropoff_month',data=df_taxi,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs each month')
plt.tight_layout()
#Plotting daily Pickup and Dropoff trip distribution
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,4))
sns.countplot(x='pickup_day',data=df_taxi,ax=ax[0])
ax[0].set_title('The distribution of number of pickups each day')
sns.countplot(x='dropoff_day',data=df_taxi,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs each day')
plt.tight_layout()
#Plotting hourly Pickup and Dropoff trip distribution
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,5))
sns.countplot(x='pickup_hour',data=df_taxi,ax=ax[0])
ax[0].set_title('The distribution of number of pickups each hour')
sns.countplot(x='dropoff_hour',data=df_taxi,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs each hour')
plt.tight_layout()
#Creating a new column according to the traffic scenerio of New York
def rush_hour(hour):
    if hour.item()>=7 and hour.item()<=9:
        return 'rush_hour_morning(7-9)'
    elif hour.item()>9 and hour.item()<16:
        return 'normal_hour_afternoon(9-16)'
    elif hour.item()>=16 and hour.item()<=19:
        return 'rush_hour_evening(16-19)'
    elif hour.item()>19 and hour.item()<=23:
        return 'normal_hour_evining(19-23)'
    else:
        return 'latenight(23 onwards)'
df_taxi['traffic_scenerio_pickup']=df_taxi[['pickup_hour']].apply(rush_hour, axis=1)
df_taxi['traffic_scenerio_dropoff']=df_taxi[['dropoff_hour']].apply(rush_hour, axis=1)
#Plotting pickup and dropoff trip distribution as per traffic scenerio
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,5))
sns.countplot(x='traffic_scenerio_pickup',data=df_taxi,ax=ax[0])
ax[0].set_title('The distribution of number of pickups as per traffics scenerio')
sns.countplot(x='traffic_scenerio_dropoff',data=df_taxi,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs as per traffics scenerio')
plt.tight_layout()
sns.distplot(df_taxi['trip_duration'],kde=True)
plt.title('The distribution of of the Pick Up  Duration distribution')
sns.boxplot(df_taxi['trip_duration'], orient='horizontal')
plt.title('A boxplot depicting the pickup duration distribution')
#Dropping trip_duration <1 min
df_taxi= df_taxi[df_taxi.trip_duration>60] # >1 min
#Dropping trip_duration >2 Hrs
df_taxi= df_taxi[df_taxi.trip_duration<=7200] # >2 hrs
sns.countplot(x='vendor_id',data=df_taxi)
#Checking Longitude and Lattitude bounds available in the data
print('Longitude Bounds: {} to {}'.format(max(df_taxi.pickup_longitude.min(),df_taxi.dropoff_longitude.min()),max(df_taxi.pickup_longitude.max(),df_taxi.dropoff_longitude.max())))
print('Lattitude Bounds: {} to {}'.format(max(df_taxi.pickup_latitude.min(),df_taxi.dropoff_latitude.min()),max(df_taxi.pickup_latitude.max(),df_taxi.dropoff_latitude.max())))
#The borders of NY City, in coordinates comes out to be: city_long_border = (-74.03, -73.75) & city_lat_border = (40.63, 40.85)
#Comparing this to our 'df_taxi.describe()' output we see that there are some coordinate points (pick ups/drop offs) that fall outside these borders. So let's limit our area of investigation to within the NY City borders.
df_taxi = df_taxi[df_taxi['pickup_longitude'] <= -73.75]
df_taxi = df_taxi[df_taxi['pickup_longitude'] >= -74.03]
df_taxi = df_taxi[df_taxi['pickup_latitude'] <= 40.85]
df_taxi = df_taxi[df_taxi['pickup_latitude'] >= 40.63]
df_taxi = df_taxi[df_taxi['dropoff_longitude'] <= -73.75]
df_taxi = df_taxi[df_taxi['dropoff_longitude'] >= -74.03]
df_taxi = df_taxi[df_taxi['dropoff_latitude'] <= 40.85]
df_taxi = df_taxi[df_taxi['dropoff_latitude'] >= 40.63]
#Getting distance(in km) from geographocal co-ordinates
from math import radians, sin, cos, sqrt, asin
def haversine(columns):
    lat1, lon1, lat2, lon2 = columns
    R = 6372.8 # Earth radius in kilometers
    
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    
    return R * c

cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
distances = df_taxi[cols].apply(lambda x: haversine(x),axis = 1)
df_taxi['distance_km'] = distances.copy()
df_taxi['distance_km'] = round(df_taxi.distance_km,2)
sns.scatterplot(x='distance_km',y='trip_duration',data=df_taxi)
#Removing distance Outliers
df_taxi = df_taxi[df_taxi['distance_km'] > 0]
#Getting Speed(Km/h) of the taxi 
df_taxi['speed_km/h']= 3600*(df_taxi.distance_km/df_taxi.trip_duration)  #3600 to convert it from km/s to km/h
#Checking Distance and Speed range
print('Distance Bounds: {} to {}'.format(df_taxi.distance_km.min(),df_taxi.distance_km.max()))
print('Speed Bounds: {} to {}'.format(df_taxi['speed_km/h'].min(),df_taxi['speed_km/h'].max()))
#Removing speed Outliers
df_taxi = df_taxi[df_taxi['speed_km/h'] > 0]
df_taxi = df_taxi[df_taxi['speed_km/h'] < 100]
#Dropping passenger count=0
df_taxi= df_taxi[df_taxi.passenger_count>0]
df_taxi['passenger_count'].value_counts()
#Plotting Trip Distribution
plt.figure(figsize=(10,6))
plt.hist(df_taxi.trip_duration, bins=100)
plt.xlabel('Trip_duration')
plt.ylabel('Number of Trips')
plt.title('Trip Distribution')
plt.show()
#Applying Feature Scaling in trip_duration caloumn to normalize the data
df_taxi['log_trip_duration']= np.log1p(df_taxi['trip_duration'])
plt.hist(df_taxi['log_trip_duration'].values, bins=100)
plt.title('Log Trip Distribution')
plt.xlabel('log(trip_duration)')
plt.ylabel('Number of Trips')
plt.show()
sns.distplot(df_taxi["log_trip_duration"], bins =100)
#Visualizing Passenger road map for picking up
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,8))
plt.ylim(40.63, 40.85)
plt.xlim(-74.03,-73.75)
ax.scatter(df_taxi['pickup_longitude'],df_taxi['pickup_latitude'], s=0.02, alpha=1)
#Visualizing Passenger road map for dropoff
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,8))
plt.ylim(40.63, 40.85)
plt.xlim(-74.03,-73.75)
ax.scatter(df_taxi['dropoff_longitude'],df_taxi['dropoff_latitude'], s=0.02, alpha=1)
#Converting Data to Geo Dataframe for pickup 
gdf=gpd.GeoDataFrame(df_taxi,geometry=gpd.points_from_xy(df_taxi['pickup_longitude'],df_taxi['pickup_latitude']))
#Geometry point has been generated for pickup
gdf.head(2)
#Visulizing pickup points with geopandas
gdf.plot(figsize=(12,10))
#Getting New York City map from Geopandas
nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
ax = nyc.plot(figsize=(12, 10))
#Applying one hot encoding to the catagorical variables
taxi_vendor=pd.get_dummies(df_taxi['vendor_id'], prefix='vendor_id',drop_first= True)
taxi_pax=pd.get_dummies(df_taxi['passenger_count'], prefix='passenger',drop_first= True)
taxi_store_and_fwd_flag=pd.get_dummies(df_taxi['store_and_fwd_flag'], prefix='store_and_fwd_flag',drop_first= True)
taxi_pickup_day=pd.get_dummies(df_taxi['pickup_day'], prefix='pickup_day',drop_first= True)
taxi_dropoff_day=pd.get_dummies(df_taxi['dropoff_day'], prefix='dropoff_day',drop_first= True)
taxi_pickup_month=pd.get_dummies(df_taxi['pickup_month'], prefix='pickup_month',drop_first= True)
taxi_dropoff_month=pd.get_dummies(df_taxi['dropoff_month'], prefix='dropoff_month',drop_first= True)
taxi_pickup_traffic_scenerio=pd.get_dummies(df_taxi['traffic_scenerio_pickup'], prefix='pickup_',drop_first= True)
taxi_dropoff_traffic_scenerio=pd.get_dummies(df_taxi['traffic_scenerio_dropoff'], prefix='dropoff_',drop_first= True)
#Adding encoded columns to final data
df_taxi=pd.concat([df_taxi,taxi_pax,taxi_vendor,taxi_store_and_fwd_flag,taxi_pickup_day,taxi_dropoff_day,taxi_pickup_month,taxi_dropoff_month,taxi_pickup_traffic_scenerio,taxi_dropoff_traffic_scenerio],axis=1)
df_taxi.head(2)
#Dropping unnecessary columns from dataset
df_taxi=df_taxi.drop(['id','vendor_id','passenger_count','pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','log_trip_duration','speed_km/h','store_and_fwd_flag','traffic_scenerio_pickup','traffic_scenerio_dropoff','pickup_month','dropoff_month','pickup_day','dropoff_day','pickup_hour','dropoff_hour','geometry','dropoff_month_7'],axis=1)
df_taxi.columns
#Assigning X and y variables
X = df_taxi.drop('trip_duration',1)
y = df_taxi['trip_duration']
#Splitting the dataset into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
#RMSE score 
lr_rmse = np.sqrt(metrics.mean_squared_error(lr_pred,y_test))
lr_rmse
#R2 score
lr_r2score = metrics.r2_score(lr_pred,y_test)
lr_r2score
#Train Score
lr_train=lr.score(X_train,y_train)
lr_train
#Test Score
lr_test=lr.score(X_test,y_test)
lr_test
#Null RMSE
y_null=np.zeros_like(y_test,dtype=float)
y_null.fill(y_test.mean())
np.sqrt(metrics.mean_squared_error(y_test,y_null))
coef1 = pd.DataFrame(lr.coef_,index=X_train.columns)
coef1.plot(kind='bar', title='Model Coefficients')
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
#RMSE score 
dt_rmse = np.sqrt(metrics.mean_squared_error(dt_pred,y_test))
dt_rmse
#R2 score
dt_r2score = metrics.r2_score(dt_pred,y_test)
dt_r2score
#Train Score
dt_train=dt.score(X_train,y_train)
dt_train
#Test Score
dt_test=dt.score(X_test,y_test)
dt_test
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
#RMSE score 
rf_rmse = np.sqrt(metrics.mean_squared_error(rf_pred,y_test))
rf_rmse
#R2 score
rf_r2score = metrics.r2_score(rf_pred,y_test)
rf_r2score
#Train Score
rf_train=rf.score(X_train,y_train)
rf_train
#Test Score
rf_test=rf.score(X_test,y_test)
rf_test
ab=AdaBoostRegressor()
ab.fit(X_train,y_train)
ab_pred=ab.predict(X_test)
#RMSE score 
ab_rmse = np.sqrt(metrics.mean_squared_error(ab_pred,y_test))
ab_rmse
#R2 score
ab_r2score = metrics.r2_score(ab_pred,y_test)
ab_r2score
#Train Score
ab_train=ab.score(X_train,y_train)
ab_train
#Test Score
ab_test=ab.score(X_test,y_test)
ab_test
gb = GradientBoostingRegressor()
gb.fit(X_train,y_train)
gb_pred = gb.predict(X_test)
#RMSE score 
gb_rmse = np.sqrt(metrics.mean_squared_error(gb_pred,y_test))
gb_rmse
#R2 score
gb_r2score = metrics.r2_score(gb_pred,y_test)
gb_r2score
#Train Score
gb_train=gb.score(X_train,y_train)
gb_train
#Test Score
gb_test=gb.score(X_test,y_test)
gb_test
xgb= XGBRegressor()
xgb.fit(X_train,y_train)
xgb_pred=xgb.predict(X_test)
#RMSE score 
xgb_rmse = np.sqrt(metrics.mean_squared_error(xgb_pred,y_test))
xgb_rmse
#R2 score
xgb_r2score = metrics.r2_score(xgb_pred,y_test)
xgb_r2score
#Train Score
xgb_train=xgb.score(X_train,y_train)
xgb_train
#Test Score
xgb_test=xgb.score(X_test,y_test)
xgb_test
lgbm = lgb.LGBMRegressor()
lgbm.fit(X_train,y_train)
lgbm_pred = lgbm.predict(X_test)
#RMSE score 
lgbm_rmse = np.sqrt(metrics.mean_squared_error(lgbm_pred,y_test))
lgbm_rmse
#R2 score
lgbm_r2score = metrics.r2_score(lgbm_pred,y_test)
lgbm_r2score
#Train Score
lgbm_train=lgbm.score(X_train,y_train)
lgbm_train
#Test Score
lgbm_test=lgbm.score(X_test,y_test)
lgbm_test
#Creating dictionary for all the metrics and models
metrics = {'Metrics': ['RMSE Score','R2 Score','Train Score','Test Score'],'Linear Regression':[lr_rmse,lr_r2score,lr_train,lr_test],
          'Decision Tree Regressor':[dt_rmse,dt_r2score,dt_train,dt_test],'Random Forest Regressor':[rf_rmse,rf_r2score,rf_train,rf_test],
        'AdaBoost Regressor':[ab_rmse,ab_r2score,ab_train,ab_test],
          'GradientBoosting Regressor':[gb_rmse,gb_r2score,gb_train,gb_test],'XGBoost Regressor':[xgb_rmse,xgb_r2score,xgb_train,xgb_test],
           'LGBM Regressor':[lgbm_rmse,lgbm_r2score,lgbm_train,lgbm_test]}
#Converting dictionary to dataframe
metrics = pd.DataFrame(metrics)
metrics
#Finding the importance of columns for prediction
perm = PermutationImportance(xgb, random_state=1).fit(X_test,xgb_pred)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())