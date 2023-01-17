import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.zip', compression='zip', header=0, sep=',', quotechar='"')
train.info()
train.head()
train.vendor_id.value_counts()
train.passenger_count.value_counts()
train.store_and_fwd_flag.value_counts()
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()

train["store_and_fwd_flag"] = l.fit_transform(train["store_and_fwd_flag"])
train.corrwith(train["trip_duration"])
train.store_and_fwd_flag.value_counts()
x_train= train[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']]
y_train = train["trip_duration"]
test = pd.read_csv("../input/nyc-taxi-trip-duration/test.zip")
test.info()
test["store_and_fwd_flag"] = l.fit_transform(test["store_and_fwd_flag"])
x_test = test[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']]
x_train.shape,x_test.shape,y_train.shape
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred = abs(y_pred)
s = pd.read_csv("../input/nyc-taxi-trip-duration/sample_submission.zip")
s.head()
f = {"id":s["id"],"trip_duration":y_pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)
f.head()