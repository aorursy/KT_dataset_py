import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2,radians
import xgboost
from sklearn.preprocessing import StandardScaler


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv",nrows=1005000)
test = pd.read_csv("../input/test.csv")
testkey = test.key
df = df.dropna(how = 'any', axis = 'rows')
len(df)
df.head()
df.describe()
l = df[(df.pickup_latitude > 42.5) | (df.pickup_latitude < 40.0)|(df.dropoff_latitude > 42.5) | (df.dropoff_latitude < 40.0) | (df.pickup_longitude > -73.0) | (df.pickup_longitude < -75.0) | (df.dropoff_longitude > -73.0) | (df.dropoff_longitude < -75.0)].index
df = df.drop(l,axis=0)
z = df[(df.fare_amount > 350.0) | (df.fare_amount < 0.0) |(df.passenger_count > 7.0) | (df.passenger_count < 0.0)].index
df = df.drop(z,axis=0)
len(df)
def distlatlong(lon1,lat1,lon2,lat2):
    lat1=radians(lat1)
    lat2=radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6373.0 * c
    return distance

df['dist'] = [distlatlong(df.pickup_longitude[i],df.pickup_latitude[i],df.dropoff_longitude[i],df.dropoff_latitude[i]) for i in df.index]
test['dist'] = [distlatlong(test.pickup_longitude[i],test.pickup_latitude[i],test.dropoff_longitude[i],test.dropoff_latitude[i]) for i in test.index]
#sns.boxplot(df.dist)
test.head()
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
df.info()
df['latenights'] = [1 if df.pickup_datetime[i].hour < 5 else 0 for i in df.index]
test['latenights'] = [1 if test.pickup_datetime[i].hour < 5 else 0 for i in test.index]
#sns.countplot('fare_amount',data = df)
df.pickup_datetime[0].weekday()
df['weekday'] = [1 if df.pickup_datetime[i].weekday() > 4 else 0 for i in df.index]
test['weekday'] = [1 if test.pickup_datetime[i].weekday() > 4 else 0 for i in test.index]

df.head()
df['year'] = [df.pickup_datetime[i].year for i in df.index]
test['year'] = [test.pickup_datetime[i].year for i in test.index]
df['day'] = [df.pickup_datetime[i].day for i in df.index]
test['day'] = [test.pickup_datetime[i].day for i in test.index]
df.head()
test.head()
#feat = df.drop(['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
#test = test.drop(['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
feat = df.drop(['key','pickup_datetime'],axis=1)
test = test.drop(['key','pickup_datetime'],axis=1)
test.year.unique()
feat = pd.concat([feat,pd.get_dummies(feat.year)],axis=1)
test = pd.concat([test,pd.get_dummies(test.year)],axis=1)
feat = feat.drop('year',axis=1)
test = test.drop('year',axis=1)
feat.head()
test.head()
label = feat['fare_amount']
feat = feat.drop('fare_amount',axis=1)
xtr,xts,ytr,yts = train_test_split(feat,label)
xgbtrain = xgboost.DMatrix(xtr,ytr)
xgbtest = xgboost.DMatrix(xts,yts)
xgbfinaltest = xgboost.DMatrix(test)
params={'eval_metric':'rmse','objective':'reg:linear'}
xgbmodel = xgboost.train(params,
                    dtrain=xgbtrain,num_boost_round=350, 
                    early_stopping_rounds=30,evals=[(xgbtest,'test')])
pred = xgbmodel.predict(xgbfinaltest)
pred = pred.round(2)
finalset = pd.DataFrame({'key':testkey,'fare_amount':pred})
finalset = finalset[['key','fare_amount']]
finalset.head()
finalset.to_csv('finaloutput.csv',index=False)
