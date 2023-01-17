import os

import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
d_train =pd.read_csv("../input/train.csv")

d_test =pd.read_csv("../input/test.csv")
d_test.columns
d_train.head(10)
d_train.describe()
# Detection des valeurs nulles ou NAN



print(d_train.isna().sum())
# Analyse univariée sur trip_duration



plt.subplots(figsize=(10,10))

plt.title("Trip_duration outliers")

d_train.boxplot(column='trip_duration')

d_train['pickup_datetime'] = pd.to_datetime(d_train['pickup_datetime'])

d_train['dropoff_datetime'] = pd.to_datetime(d_train['dropoff_datetime'])
d_train1 = d_train[d_train['passenger_count']>= 1] 



d_train1 = d_train[d_train['trip_duration']>= 200 ]

d_train1 = d_train[d_train['trip_duration']<= 5000]
X=d_train1[['vendor_id','passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']]
y=d_train1['trip_duration']
d_train1.head(10)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
prediction = rf.predict(X_test)

prediction
submis_smp = pd.read_csv("../input/sample_submission.csv")

submis_smp.head()
prediction_test = rf.predict(d_test[['vendor_id','passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']])



prediction_test

submission = pd.DataFrame({'id': d_test.id, 'trip_duration': prediction_test})



submission.head()
submission.to_csv('submision.csv', index=False)