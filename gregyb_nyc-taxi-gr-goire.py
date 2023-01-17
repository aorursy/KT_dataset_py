# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import calendar

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import check_output

from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!ls ../input
!head -n 5 ../input/nyc-taxi-trip-duration/sample_submission.csv
data_f1= pd.read_csv("../input/nyc-taxi-trip-duration/train.csv")

test= pd.read_csv("../input/nyc-taxi-trip-duration/test.csv")

data_f1.shape
weather= pd.read_csv("../input/weather-nyc/KNYC_Metars.csv")

weather.shape
test.head()
data_f1.set_index("id", inplace=False)

test= test.set_index("id", inplace=False)
weather.head()
data_f1.info()
weather.info()
data_f1.describe()
data_f1.isna().sum()
test.head()
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'],format='%Y-%m-%d %H:%M:%S')

data_f1.head()
data_f1['pickup_datetime']=pd.to_datetime(data_f1['pickup_datetime'],format='%Y-%m-%d %H:%M:%S')

data_f1['dropoff_datetime']=pd.to_datetime(data_f1['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S')

data_f1.head()
data_f1["pickup_datetime_month"]= data_f1["pickup_datetime"].dt.month

data_f1["dropoff_datetime_month"]= data_f1["dropoff_datetime"].dt.month
test['pickup_day_of_week']=test['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
data_f1['pickup_day_of_week']=data_f1['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])

data_f1['drop_off_day_of_week']=data_f1['dropoff_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
print(data_f1["trip_duration"].mean())



print(data_f1["trip_duration"].min())



print(data_f1["trip_duration"].max())
one = data_f1.groupby('vendor_id').count()['id'].iloc[0:1]

two = data_f1.groupby('vendor_id').count()['id'].iloc[1:2]
data_f1.groupby('vendor_id').count()['id'].hist();
data_f1['passenger_count'].value_counts(normalize=True).plot(kind="pie", label= "number of passenger");
data_f1["trip_duration_time"]= data_f1['dropoff_datetime']-data_f1['pickup_datetime']
plt.figure(figsize=(30,5))

sns.distplot(np.log(data_f1['trip_duration'].values))

plt.title("trip duration on second")
data_f1['distance_long'] = data_f1['pickup_longitude'] - data_f1['dropoff_longitude']

test['distance_long'] = test['pickup_longitude'] - test['dropoff_longitude']



data_f1['distance_lat'] = data_f1['pickup_latitude'] - data_f1['dropoff_latitude']

test['distance_lat'] = test['pickup_latitude'] - test['dropoff_latitude']



data_f1['distance'] = np.sqrt(np.square(data_f1['distance_long']) + np.square(data_f1['distance_lat']))

test['distance'] = np.sqrt(np.square(test['distance_long']) + np.square(test['distance_lat']))

data_f1['distance'].mean()
city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)

fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True);

ax.scatter(data_f1['pickup_longitude'], data_f1['pickup_latitude'],color='green')

ax.set_ylabel('Latitude')

ax.set_xlabel('Longitude')

plt.ylim(city_lat_border)

plt.xlim(city_long_border)

plt.title("map of taxi in new york")
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor
test.head()
data_f1.set_index('id', inplace=True)

data_f1.head()
data_f1.drop(['pickup_datetime', 'dropoff_datetime', 

              'pickup_datetime_month', 'trip_duration_time',

              'dropoff_datetime_month', 'drop_off_day_of_week'], axis=1, inplace=True)

data_f1.info()
test.drop(['pickup_datetime'], axis=1, inplace=True)
test.info()
data_f1.info()
for c in data_f1.select_dtypes('object').columns:

    data_f1[c] = data_f1[c].astype('category').cat.codes



data_f1.info()
for c in test.select_dtypes('object').columns:

    test[c] = test[c].astype('category').cat.codes



test.info()
X_train = data_f1.drop('trip_duration', axis=1)

y_train = data_f1['trip_duration']
rf = RandomForestRegressor(n_estimators=10, random_state=42)
X_train.head()


y_train = np.log1p(y_train)
#cvscore =  cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')

#np.mean(np.sqrt(-cvscore))
#print('cross_val_score average: ', np.mean(np.sqrt(-cvscore)))
%%time

rf.fit(X_train, y_train)
valid_pred_rf = rf.predict(test)

valid_pred_rf = np.expm1(valid_pred_rf)
valid_pred_rf_df = pd.DataFrame(valid_pred_rf, index=test.index)

valid_pred_rf_df.columns = ['trip_duration']

valid_pred_rf_df.to_csv('submit_file.csv')

pd.read_csv('submit_file.csv').head()