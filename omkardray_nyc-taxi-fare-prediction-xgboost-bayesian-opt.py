# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd



df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 10_000_000)

df.dtypes
df.head(10)
df.shape
df.describe()
df.dropna()

df.shape
df.isnull().sum().sort_values(ascending=False)
df = df.drop(df[df['fare_amount']<0].index, axis=0)

df.shape
df.head(15)
df['fare_amount'].sort_values(ascending=False)
df = df.drop(df[df['fare_amount']==0].index, axis=0)

df.shape
print('Old size: %d' % len(df))

df = df.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(df))
df.columns
df['passenger_count'].describe()
import matplotlib.pyplot as plt

df.passenger_count.hist(bins=200, figsize=(20,3))

plt.title('Histogram');
df.passenger_count.describe
df.head()
df=df.drop(df[df['passenger_count']>5].index,axis=0)

df.shape
df['passenger_count'].describe()

passenger_count = df['passenger_count']

plt.hist(passenger_count)

plt.show()



df['pickup_latitude'].describe()
df = df.drop((df[df['dropoff_latitude']<-90]).index, axis=0)

df = df.drop((df[df['dropoff_latitude']>90]).index, axis=0)

df = df.drop((df[df['pickup_latitude']<-90]).index, axis=0)

df = df.drop((df[df['pickup_latitude']>90]).index, axis=0)

df.shape
df = df.drop((df[df['dropoff_longitude']<-180]).index, axis=0)

df = df.drop((df[df['dropoff_longitude']>180]).index, axis=0)

df = df.drop((df[df['pickup_longitude']<-180]).index, axis=0)

df = df.drop((df[df['pickup_longitude']>180]).index, axis=0)

df.shape
df.dtypes
df.head()
df.dtypes
from datetime import datetime

df['key'] = pd.to_datetime(df['key'])

df.head()



df.dtypes
df.loc[:, 'year']= df['key'].dt.year

df.loc[:, 'month']= df['key'].dt.month

df.head()

df.loc[:, 'hour']= df['key'].dt.hour

df.loc[:, 'Day of Week']= df['key'].dt.dayofweek

df.head()
df.corr()
plt.figure(figsize=(15,7))

plt.scatter(x=df['key'], y=df['fare_amount'], s=1.5)

plt.xlabel('Date')

plt.ylabel('Fare')

plt.show()
def haversine_distance(lat1, long1, lat2, long2):

    data = [df]

    for i in data:

        R = 6371  #radius of earth in kilometers

        #R = 3959 #radius of earth in miles

        phi1 = np.radians(i[lat1])

        phi2 = np.radians(i[lat2])

    

        delta_phi = np.radians(i[lat2]-i[lat1])

        delta_lambda = np.radians(i[long2]-i[long1])

    

        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    

        #c = 2 * atan2( √a, √(1−a) )

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    

        #d = R*c

        d = (R * c) #in kilometers

        i['H_Distance'] = d

    return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df.head()
plt.figure(figsize=(15,7))

plt.scatter(x=df['H_Distance'], y=df['fare_amount'], s=15.5)

plt.xlabel('Distance')

plt.ylabel('Fare')

plt.show()
df.sort_values(['H_Distance','fare_amount'], ascending=False)
df.corr()
df.describe()
high_distance = df.loc[(df['H_Distance']>200)&(df['fare_amount']!=0)]

high_distance.shape
high_distance.describe()
df.groupby('passenger_count')['H_Distance','fare_amount'].mean()
df=df.drop(df[df['passenger_count']==0].index,axis=0)

df.shape
df.groupby('passenger_count')['H_Distance','fare_amount'].mean()
df.head()
import seaborn as sns

plt.figure(figsize = (10, 6))

sns.distplot(df['fare_amount']);

plt.title('Distribution of Fare');
df.head()
df.columns
df.passenger_count.unique()
def generate_features(df):

    

    aggs={}

    aggs['month']= ['nunique','mean']

    aggs['year']= ['nunique']

    aggs['hour']= ['nunique','mean']

    aggs['Day of Week']=['nunique','mean']

    aggs['passenger_count']=['nunique']

    aggs['fare_amount']=['mean']

    agg_df=df.groupby('key').agg(aggs)

    agg_df = agg_df.reset_index()

    

    return agg_df

    
print(generate_features(df))
generate_features(df).head()
df.head()
df.dtypes
from sklearn import preprocessing



lbl_enc = preprocessing.LabelEncoder()

df.loc[:,'year']=lbl_enc.fit_transform(df.year.values)
df.head()
from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection



X= df[['passenger_count','year','month','hour','Day of Week','H_Distance']].values

y= df.fare_amount.values
reg = ensemble.RandomForestRegressor()



param_grid = {

    'n_estimators':[100,200,300,400,500],

    'max_depth' : [1,2,5,7,11,15],

    'criterion': ['mse','mae']

}