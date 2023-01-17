# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import Counter

from sklearn.ensemble import RandomForestRegressor

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/train.csv",nrows=1000000)

test_df=pd.read_csv("../input/test.csv")
train_df.shape
train_df.columns
train_df.head()
train_df.info()
train_df.describe()
test_df.info()
test_df.describe()
train_df.isnull().sum()
#Drop the missing values record

train_df= train_df.drop(train_df[train_df.isnull().any(1)].index, axis = 0)
train_df.info()
Counter(train_df['fare_amount']<0)
#Drop the negative fare record

train_df= train_df.drop(train_df[train_df['fare_amount']<0].index, axis = 0)

train_df.shape
train_df.describe()
Counter(train_df['passenger_count']>6)
train_df= train_df.drop(train_df[train_df['passenger_count']>6].index, axis = 0)

train_df.shape
Counter(train_df['pickup_latitude']<-90)
Counter(train_df['pickup_latitude']>90)
#We need to drop these outliers

train_df = train_df.drop(((train_df[train_df['pickup_latitude']<-90])|(train_df[train_df['pickup_latitude']>90])).index, axis=0)
train_df.shape
Counter(train_df['pickup_longitude']<-180)
Counter(train_df['pickup_longitude']>180)
#We need to drop these outliers

train_df = train_df.drop((train_df[train_df['pickup_longitude']<-180]).index, axis=0)
train_df.shape
train_df.dtypes
train_df.head(3)
train_df['key']=pd.to_datetime(train_df['key'])

train_df['pickup_datetime']=pd.to_datetime(train_df['pickup_datetime'])
train_df.dtypes
test_df.dtypes
train_df.head()
test_df['key']=pd.to_datetime(test_df['key'])

test_df['pickup_datetime']=pd.to_datetime(test_df['pickup_datetime'])
test_df.dtypes
test_df.head()
train_df.head()
data=[train_df,test_df]

for i in data:

    i['date']=i['pickup_datetime'].dt.day

    i['month']=i['pickup_datetime'].dt.month

    i['day_of_week']=i['pickup_datetime'].dt.dayofweek

    i['hour']=i['pickup_datetime'].dt.hour

    i['year']=i['pickup_datetime'].dt.year

    
train_df.head()
train_df.describe()
def sphere_distance(lat1,long1,lat2,long2):

    data=[train_df,test_df]

    for i in data:

        R=6367

        phi1 = np.radians(i[lat1])

        phi2 = np.radians(i[lat2])

        delta_phi = np.radians(i[lat2]-i[lat1])

        delta_lambda = np.radians(i[long2]-i[long1])

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        d = (R * c)

        i['S_Distance'] = d

    return d #in Kilometer
sphere_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
train_df.head()
plt.hist(train_df['passenger_count'], bins=15)

plt.xlabel('No. of Passengers')

plt.ylabel('Frequency')
plt.scatter(x=train_df['passenger_count'], y=train_df['fare_amount'],s=2.0)

plt.xlabel('No. of Passengers')

plt.ylabel('Fare')
plt.scatter(x=train_df['date'], y=train_df['fare_amount'])

plt.xlabel('Date')

plt.ylabel('Fare')
plt.hist(train_df['hour'],bins=50)

plt.xlabel('Date')

plt.ylabel('Fare')
plt.hist(train_df['day_of_week'],bins=20)

plt.xlabel('Date')

plt.ylabel('Fare')
plt.scatter(x=train_df['day_of_week'], y=train_df['fare_amount'])

plt.xlabel('Date of week')

plt.ylabel('Fare')
len(train_df)
train_df.sort_values(['S_Distance','fare_amount'], ascending=False)
dis_0 = train_df.loc[(train_df['S_Distance'] == 0), ['S_Distance']]

dis_1 = train_df.loc[(train_df['S_Distance'] > 0) & (train_df['S_Distance'] <= 10), ['S_Distance']]

dis_2 = train_df.loc[(train_df['S_Distance'] > 10) & (train_df['S_Distance'] <= 50), ['S_Distance']]

dis_3 = train_df.loc[(train_df['S_Distance'] > 50) & (train_df['S_Distance'] <= 100), ['S_Distance']]

dis_4 = train_df.loc[(train_df['S_Distance'] > 100) & (train_df['S_Distance'] <= 200), ['S_Distance']]

dis_5 = train_df.loc[(train_df['S_Distance'] > 200) & (train_df['S_Distance'] <= 300), ['S_Distance']]

dis_6 = train_df.loc[(train_df['S_Distance'] > 300) & (train_df['S_Distance'] <= 500), ['S_Distance']]

dis_7 = train_df.loc[(train_df['S_Distance'] > 500), ['S_Distance']]

dis_0['bins']='0'

dis_1['bins']='0-10'

dis_2['bins']='11-50'

dis_3['bins']='51-100'

dis_4['bins']='101-200'

dis_5['bins']='201-300'

dis_6['bins']='301-500'

dis_7['bins']='>500'

dis_bin=pd.concat([dis_0,dis_1,dis_2,dis_3,dis_4,dis_5,dis_6,dis_7])

dis_bin
x=Counter(dis_bin['bins'])

x
train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)]
train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)]
train_df = train_df.drop(train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)].index, axis=0)
train_df.shape
#dropoff latitude and longitude = 0

train_df = train_df.drop(train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)].index, axis=0)

train_df.shape
high_distance = train_df.loc[(train_df['S_Distance']>200)&(train_df['fare_amount']!=0)]
high_distance
high_distance.shape
high_distance['S_Distance'] = high_distance.apply(

    lambda row: (row['fare_amount'] - 2.50)/1.56,

    axis=1

)
high_distance
train_df.update(high_distance)
train_df
train_df[train_df['S_Distance']==0]
train_df[(train_df['S_Distance']==0)&(train_df['fare_amount']==0)]
train_df = train_df.drop(train_df[(train_df['S_Distance']==0)&(train_df['fare_amount']==0)].index, axis = 0)
#Between 6AM and 8PM on Mon-Fri

rush_hour = train_df.loc[(((train_df['hour']>=6)&(train_df['hour']<=20)) & ((train_df['day_of_week']>=1) & (train_df['day_of_week']<=5)) & (train_df['S_Distance']==0) & (train_df['fare_amount'] < 2.5))]

rush_hour
train_df=train_df.drop(rush_hour.index,axis=0)
train_df.shape
non_rush_hour = train_df.loc[(((train_df['hour']<6)|(train_df['hour']>20)) & ((train_df['day_of_week']>=1)&(train_df['day_of_week']<=5)) & (train_df['S_Distance']==0) & (train_df['fare_amount'] < 3.0))]
non_rush_hour
non_rush_hour = train_df.loc[(((train_df['hour']<6)|(train_df['hour']>20)) & ((train_df['day_of_week']>=1)&(train_df['day_of_week']<=5)) & (train_df['S_Distance']==0) & (train_df['fare_amount'] < 3.0))]

non_rush_hour
train_df.loc[(train_df['S_Distance']!=0) & (train_df['fare_amount']==0)]
scenario_3 = train_df.loc[(train_df['S_Distance']!=0) & (train_df['fare_amount']==0)]

scenario_3
scenario_3 = train_df.loc[(train_df['S_Distance']!=0) & (train_df['fare_amount']==0)]
scenario_3['fare_amount'] = scenario_3.apply(

    lambda row: ((row['S_Distance'] * 1.56) + 2.50), axis=1

)
scenario_3['fare_amount']
train_df.loc[(train_df['S_Distance']==0) & (train_df['fare_amount']!=0)]
scenario_4 = train_df.loc[(train_df['S_Distance']==0) & (train_df['fare_amount']!=0)]
scenario_4
len(scenario_3)
len(scenario_4)
scenario_4.loc[(scenario_4['fare_amount']<=3.0)&(scenario_4['S_Distance']==0)]
scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['S_Distance']==0)]
scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['S_Distance']==0)]
len(scenario_4_sub)
scenario_4_sub['S_Distance'] = scenario_4_sub.apply(

lambda row: ((row['fare_amount']-2.50)/1.56), axis=1

)
train_df.update(scenario_4_sub)
len(train_df)
train_df.columns
test_df.columns
train_df = train_df.drop(['key','pickup_datetime'], axis = 1)

test_df = test_df.drop(['key','pickup_datetime'], axis = 1)
train_df.columns
test_df.columns
x_train = train_df.iloc[:,train_df.columns!='fare_amount']

y_train = train_df['fare_amount'].values

x_test = test_df
x_train.shape
y_train.shape
rg=RandomForestRegressor()

rg.fit(x_train,y_train)

y_predict=rg.predict(x_test)

y_predict
submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = y_predict

submission.to_csv('submission_1.csv', index=False)

submission.head(10)