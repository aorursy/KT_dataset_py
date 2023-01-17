import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import os

# print(os.listdir("../input"))

%matplotlib inline

sns.set({'figure.figsize':(16,8)})
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")

#train_data=pd.read_csv("/Users/mbp/Desktop/HETIC/KAGGLE/nyc-taxi-trip-duration/train.csv")

#test_data=pd.read_csv("/Users/mbp/Desktop/HETIC/KAGGLE/nyc-taxi-trip-duration/test.csv")



train_data.describe()
train_data.head()
print(test_data.shape)

print(train_data.shape)
print(f"Data Frame a {train_data.shape[0]} lignes et {train_data.shape[1]} colonnes")

print(f"La colonne d'id a {train_data.id.nunique()} de la valeur identique")

train_data.duplicated().sum()
train_data.isna().sum()
fig, ax = plt.subplots(7, sharex=True)

for i,c in enumerate(["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","trip_duration"]):

    sns.boxplot(train_data[c],ax=ax[i],width=1.5)

    ax[i].set_xscale("log")

    ax[i].set_xlabel("")

    ax[i].set_ylabel(c, fontsize=15,rotation=45)

fig.suptitle('Analyse des outliers', fontsize=20)

train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])

train_data['dropoff_datetime'] = pd.to_datetime(train_data['dropoff_datetime'])

train_data.info()
train_data["store_and_fwd_flag"]=np.where(train_data["store_and_fwd_flag"] == "N",0,1)

train_data.head()
X=train_data[['vendor_id',

       'passenger_count', 'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude']]

X.head()
y=train_data['trip_duration']

y.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score;
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

rf = RandomForestRegressor(n_estimators=70)

rf.fit(X_train,y_train)
pred = rf.predict(X_test)

pred
from sklearn.metrics import mean_squared_log_error 
print(mean_squared_log_error(y_test,pred))
submission =pd.read_csv("../input/sample_submission.csv")

#submission =pd.read_csv("/Users/mbp/Desktop/HETIC/KAGGLE/nyc-taxi-trip-duration/train.csv")

submission.head()
test_pred = rf.predict(test_data[['vendor_id',

       'passenger_count', 'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude']])

print(test_pred)
submission['trip_duration'] = test_pred
submission.head()
# my_submission.to_csv('/Users/mbp/Desktop/HETIC/KAGGLE/nyc-taxi-trip-duration/submission.csv', index=False)

submission.to_csv('submission_file.csv', index=False)