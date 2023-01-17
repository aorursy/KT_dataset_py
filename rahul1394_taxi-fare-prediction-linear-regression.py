import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings  

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error
import datashader as ds
taxi = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',nrows=50000)

taxi.head()
taxi.info()
taxi.describe()
taxi.isnull().sum()
taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])
taxi['year'] = taxi['pickup_datetime'].dt.year
taxi['month'] = taxi['pickup_datetime'].dt.month
taxi['hour'] = taxi['pickup_datetime'].dt.hour
taxi['day_of_week'] = taxi['pickup_datetime'].dt.dayofweek
sns.boxplot(taxi['fare_amount'])
sns.boxplot(taxi['pickup_latitude'])
sns.boxplot(taxi['pickup_longitude'])
sns.boxplot(taxi['dropoff_latitude'])
sns.boxplot(taxi['dropoff_longitude'])
sns.boxplot(taxi['passenger_count'])
sns.boxplot(taxi['year'])
sns.boxplot(taxi['day_of_week'])
def iqr(col):

    q1,q3 = np.quantile(taxi[col],[0.25,0.75])

    iqR = q3 - q1

    ul = q3 + 1.5*iqR

    ll = q1 - 1.5*iqR

    taxi[col] = taxi[(taxi[col] <= ul) & (taxi[col] >= ll)][col]
#iqr('passenger_count')
iqr('dropoff_longitude')
iqr('dropoff_latitude')
iqr('pickup_longitude')
iqr('pickup_latitude')
iqr('fare_amount')
taxi['fare_amount'].fillna(taxi['fare_amount'].mean(),inplace=True)
taxi['pickup_latitude'].fillna(method='ffill',inplace=True)
taxi['pickup_longitude'].fillna(method='ffill',inplace=True)
taxi['dropoff_latitude'].fillna(method='ffill',inplace=True)
taxi['dropoff_longitude'].fillna(method='ffill',inplace=True)
taxi.dropna(inplace=True)
def calculateDistance(plat,dlat,plong,dlong):

    

    lat1,lat2,long1,long2 = map(np.radians,[plat,dlat,plong,dlong])

    diffLat = lat2 - lat1

    diffLong = long2 - long1

    r = 6371000

    

    a = (np.sin(diffLat/2))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(diffLong/2)**2

    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))

    diskm = (r*c)/1000

    return diskm
taxi['travel_distance(km)'] = calculateDistance(taxi.pickup_latitude,taxi.dropoff_latitude,taxi.pickup_longitude,taxi.dropoff_longitude)
taxi.drop(['pickup_datetime'],axis=1,inplace=True)
taxi.head()
sns.countplot(taxi.passenger_count)
sns.heatmap(taxi.corr(),annot=True)
print(taxi.pickup_latitude.min(),taxi.pickup_latitude.max())

print(taxi.pickup_longitude.min(),taxi.pickup_longitude.max())

print(taxi.dropoff_latitude.min(),taxi.dropoff_latitude.max())

print(taxi.dropoff_longitude.min(),taxi.pickup_longitude.max())
def plotLocation(lat,long,colormap):

    x_range, y_range = ((40.686192, 40.818789),(-74.032472, -73.92981))

    cvs = ds.Canvas( x_range = x_range, y_range = y_range)

    agg = cvs.points(x=lat,y=long,source=taxi)

    img = ds.transfer_functions.shade(agg, cmap = colormap)

    return ds.transfer_functions.set_background(img,'black')
plotLocation('pickup_latitude','pickup_longitude',ds.colors.viridis)
plotLocation('dropoff_latitude','dropoff_longitude',ds.colors.Hot)
x = taxi.drop(['key','fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)

y = taxi['fare_amount']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30)
xtrain.shape,ytrain.shape,xtest.shape,ytest.shape
lr = LinearRegression()
lr.fit(xtrain,ytrain)

ypred = lr.predict(xtest)
r2_score(ytest,ypred)
mean_squared_error(ytest,ypred)**0.5