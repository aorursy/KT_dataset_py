# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", nrows=10_000_000)
test = pd.read_csv("../input/test.csv")
train.head()
train.describe()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.distplot(train.passenger_count)
test.shape
# Training set
clean_train = train[abs(train["pickup_latitude"]) < 90]
clean_train = clean_train[abs(clean_train["dropoff_latitude"]) < 90]
clean_train = clean_train[abs(clean_train["pickup_longitude"]) < 180]
clean_train = clean_train[abs(clean_train["dropoff_longitude"]) < 180]

clean_train = clean_train[clean_train["passenger_count"] < 10]
clean_train = clean_train[clean_train["fare_amount"] > 0]
clean_train = clean_train[clean_train["fare_amount"] < 500]

clean_test = test
num_missing = clean_train.isnull().sum()
num_missing
clean_train["pickup_datetime"] = pd.to_datetime(clean_train["pickup_datetime"])
clean_test["pickup_datetime"] = pd.to_datetime(clean_test["pickup_datetime"])
from math import radians, cos, sin, asin, sqrt
from numpy import arcsin

def haversine(row):
    lon1 = row['pickup_longitude']
    lat1 = row['pickup_latitude']
    lon2 = row['dropoff_longitude']
    lat2 = row['dropoff_latitude']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arcsin(sqrt(a)) 
    km = 6367 * c
    return km
# Train
clean_train['distance'] = clean_train.apply(lambda row: haversine(row), axis=1)
clean_train = clean_train.drop(["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], axis=1)
clean_train["pickup_hour"] = clean_train["pickup_datetime"].dt.hour
clean_train["pickup_day"] = clean_train["pickup_datetime"].dt.day
clean_train["pickup_month"] = clean_train["pickup_datetime"].dt.month
# Test
clean_test['distance'] = clean_test.apply(lambda row: haversine(row), axis=1)
clean_test = clean_test.drop(["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], axis=1)
clean_test["pickup_hour"] = clean_test["pickup_datetime"].dt.hour
clean_test["pickup_day"] = clean_test["pickup_datetime"].dt.day
clean_test["pickup_month"] = clean_test["pickup_datetime"].dt.month
clean_train.describe()
clean_test.describe()
clean_train = clean_train[clean_train["distance"] < 200]
clean_train = clean_train[clean_train["distance"] > 0]
sns.distplot(clean_test.distance)
clean_train["distance"] = np.log(clean_train.distance + 1)
clean_test["distance"] = np.log(clean_test.distance + 1)
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

X_all = clean_train.drop(["fare_amount", "pickup_datetime", "key"],axis=1)
y_all = clean_train["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=20)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
print("RMSE:\t$%.2f" % np.sqrt(mean_absolute_error(y_test, xgb_predict)))
print('Variance score: %.2f' % r2_score(y_test, xgb_predict))
feature_imp = pd.Series(model.feature_importances_,index=list(X_train)).sort_values(ascending=False)
feature_imp
final_test = clean_test.drop(["key", "pickup_datetime"], axis=1)
final_pred=xgb_model.predict(final_test)
submission = pd.DataFrame({'key':clean_test['key'],'fare_amount':final_pred})

#Visualize the first 5 rows
submission.head()
filename = 'FareAmountPreds.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)