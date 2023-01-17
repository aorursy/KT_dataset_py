# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express  as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',nrows=1000000)
df_test=pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')
df_train.head()
df_train.describe()
#we can see that fare is in negative, and also according to google max and min lat is [90,-90]
#and long is[180,-180].So we will remove the data points /outliers 
#we will check the dist plot of fare
(df_train['fare_amount'].hist(bins=50))
#data dist is skewed
#according to passengers_counts max is 208,lets check it

a=df_train[df_train['passenger_count']<=6]
a
# mostly its an error, we will remove this also
sns.countplot(a['passenger_count'])
#1 passengers are more, followed  by 2 and 5
#df_train[(df_train['pickup_latitude']<=90) & (df_train['pickup_latitude']>=-90) ]
train=df_train[(df_train['pickup_longitude'].between(-180,180)) & (df_train['pickup_latitude'].between(-90,90))]
train=train[train['fare_amount']>=0]#amt cannot be negative
train=train[(train['dropoff_longitude'].between(-180,180)) &(train['dropoff_latitude'].between(-90,90))]


test=df_test[(df_test['pickup_longitude'].between(-180,180)) & (df_test['pickup_latitude'].between(-90,90))]
#test=test[test['fare_amount']>=0]
test=test[(test['dropoff_longitude'].between(-180,180)) &(test['dropoff_latitude'].between(-90,90))]

train.describe()
#passenger count is 208,lets check it
train[train['passenger_count']==208]#its noise data, we will remove it
train=train[train['passenger_count']<=6]
test=test[test['passenger_count']<=6]

train.head()
train.info()
#train['key']=pd.to_datetime(train['key'])#its just unique string in both train and test.
train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])
test.info()
train['day']=train['pickup_datetime'].dt.day
train['month']=train['pickup_datetime'].dt.month
train['year']=train['pickup_datetime'].dt.year
train['hour']=train['pickup_datetime'].dt.hour
train['dayofweek']=train['pickup_datetime'].dt.dayofweek

#test
test['day']=test['pickup_datetime'].dt.day
test['month']=test['pickup_datetime'].dt.month
test['year']=test['pickup_datetime'].dt.year
test['hour']=test['pickup_datetime'].dt.hour
test['dayofweek']=test['pickup_datetime'].dt.dayofweek
test.head()
import numpy as np

def haversine(df):
    
    
    lat1= np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    #### Based on the formula  x1=drop_lat,x2=dropoff_long 
    dlat = np.radians(df['dropoff_latitude']-df["pickup_latitude"])
    dlong = np.radians(df["dropoff_longitude"]-df["pickup_longitude"])
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



#for i in range(len(t))
train['dist']=haversine(train)
test['dist']=haversine(test)
test.head()
##)
train['dist'].describe()
# there is huge diff between 75% and max 
test['dist'].describe()
train[train['dist']>=100]
# we can see that either the pickup lat/long is zero or dropoff lat/long is 0
train[(((train['pickup_latitude']==0)|(train['pickup_longitude']==0))&((train['dropoff_latitude']==0)|(train['dropoff_longitude']==0)))].head()

#dist is large bcz either lat or long is not avialable,
#fare amt is also less even though dist is very large, this is noise , we can impute dist value for which fare amt is not zero
#so we will drop rows that have both lat and long 0 for pickup and dropoff
#pickup 0
trn=train.copy()
a=trn[(((trn['pickup_latitude']==0)&(trn['pickup_longitude']==0))&((trn['dropoff_latitude']==0)|(trn['dropoff_longitude']==0)))].index
trn.drop(a,axis=0,inplace=True)

tst=test.copy()
tst[(((tst['pickup_latitude']==0)&(tst['pickup_longitude']==0)))]
#no data is present

#dropping row which has fare_amt=0 and also lat/long=0
b=trn[((trn['pickup_latitude']==0)&(trn['pickup_longitude']==0))&((trn['dropoff_latitude']!=0)|(trn['dropoff_longitude']!=0))&(trn['fare_amount']==0)].index
trn.drop(b,axis=0,inplace=True)

#no data for test

#vice versa 
b=trn[((trn['pickup_latitude']!=0)&(trn['pickup_longitude']!=0))&((trn['dropoff_latitude']==0)|(trn['dropoff_longitude']==0))&(trn['fare_amount']==0)].index
trn.drop(b,axis=0,inplace=True)

#no data for test
trn.describe()
#fareamt and dist have min 0
tst.describe()
#same lat and long for pickup and dropoff hence zero fare and  dist, drop them
same=trn[(trn['fare_amount']==0) & (trn['dist']==0) & (trn['pickup_latitude']==trn['dropoff_latitude'])].index
trn.drop(same,axis=0,inplace=True)




fare_up=trn[(trn['fare_amount']==0) & (trn['dist']!=0)]

#we wil impute value,accrding to google, on weekend initial charge=3$ and 1.5$/km and night
# we wil impute value,accrding to google, on weekdays initial charge=2.5$ and 1.5$/km
#fare=initial+dist*1.5$
#so dist=(fare-initial)/1.5

#Mon_friday Morning 6am-8pm 
fare_up_mor=fare_up[fare_up['hour'].between(6,19,inclusive=True) & (fare_up['dayofweek'].between(1,5))]
fare_up_mor['fare_amount']=fare_up.apply(lambda x : 2.5+(fare_up['dist']*1.5))
fare_up.update(fare_up_mor)

#Mon-Friday 8pm-6pm
fare_up_night=fare_up[((fare_up['hour']<6) | (fare_up['hour']>=20)) & (fare_up['dayofweek'].between(1,5))]
fare_up_night['fare_amount']=fare_up.apply(lambda x : 3+(fare_up['dist']*1.5))
fare_up.update(fare_up_night)

#saturday and sunday all day
fare_up_wkend=fare_up[(fare_up['dayofweek']==0) | (fare_up['dayofweek']==6)]
fare_up_wkend['fare_amount']=fare_up.apply(lambda x : 3+(fare_up['dist']*1.5))
fare_up.update(fare_up_wkend)


trn.update(fare_up)
trn[(trn['fare_amount']!=0) & (trn['dist']==0)]
#seems dist value does not go accordingly with fare , so we will try to impute for those datapts for which price is too high and dist traveeled is too low
dist_up=trn[(trn['fare_amount']>100) & (trn['dist']<5)]

##Mon_friday Morning 6am-8pm 
dist_up_mor=dist_up[dist_up['hour'].between(6,19,inclusive=True) & (dist_up['dayofweek'].between(1,5))]
dist_up_mor['dist']=dist_up.apply(lambda x :((dist_up['fare_amount']-2.5)/1.5))
dist_up.update(dist_up_mor)

#Mon-Friday 8pm-6pm
dist_up_night=dist_up[((dist_up['hour']<6) | (dist_up['hour']>=20)) & (dist_up['dayofweek'].between(1,5))]
dist_up_night['dist']=dist_up.apply(lambda x : ((dist_up['fare_amount']-2.5)/1.5))
dist_up.update(dist_up_night)

#saturday and sunday all day
dist_up_wkend=dist_up[(dist_up['dayofweek']==0) | (dist_up['dayofweek']==6)]
dist_up_wkend['dist']=dist_up.apply(lambda x : ((dist_up['fare_amount']-2.5)/1.5))
dist_up.update(dist_up_wkend)


trn.update(dist_up)
trn.describe()
#1> does dist  affects fare
sns.scatterplot(train[train['dist']<100]['dist'],train['fare_amount'])

#we can see that there is some linearity
#2>fare price wrt to hours
plt.figure(figsize=(20,6))
sns.barplot(train['hour'],train[train['fare_amount']<100]['fare_amount'])
#fareamt wrt to weekdays and weekends
plt.figure(figsize=(20,6))
sns.boxplot(train['dayofweek'],train[train['fare_amount']<100]['fare_amount'])
plt.figure(figsize=(20,6))
sns.barplot(train['year'],train[train['fare_amount']<100]['fare_amount'])

#Mean priec has increased over the year
trn.groupby(['month'])['passenger_count'].count().sort_values(ascending=False).plot(kind='bar')
#During first 6 months  most people availing cab
plt.figure(figsize=(15,5))
sns.countplot(trn['hour'])# most person avail cab in the evening and least in midnight
trn.describe()
#so if we look at test data, max dist is 99 and our trn data its 12594, 
ind=trn[trn['dist']>100].sort_values(by='dist',ascending=False).index

trn.loc[ind]

# for now we can will drop this

trn=trn.drop(ind,axis=0)
maxprice=trn['fare_amount'].sort_values(ascending=False).index
trn.loc[maxprice]
plt.figure(figsize=(10,7))
sns.heatmap(trn.corr(),annot=True)
#modelling

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(f'trains shape{trn.shape}  test shape{tst.shape}')
X=trn[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'day', 'month', 'year', 'hour', 'dayofweek', 'dist']]
y=trn['fare_amount']
x_train,x_test,val_train,val_test=train_test_split(X,y,test_size=0.3)
lr=LinearRegression()
##def fit(x_train,val_train,)
lr.fit(x_train,val_train)
y_hat=lr.predict(x_test)
print('R2 value is',r2_score(val_test,y_hat))
print('MAE',mean_absolute_error(val_test,y_hat))
print('SMAE',mean_squared_error(val_test,y_hat)**0.5)

test.columns
test=test[['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'day',
       'month', 'year', 'hour', 'dayofweek', 'dist']]

dt=DecisionTreeRegressor()

dt.fit(x_train,val_train)
y_hat=dt.predict(test)

submission = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
submission['fare_amount'] = y_hat
submission.to_csv('submission_1.csv', index=False)
