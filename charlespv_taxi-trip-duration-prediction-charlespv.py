import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# ML libraries

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(f"training set shape : {train.shape}")

print(f"testing set shape : {test.shape}")
train.head(5)
test.head(5)
train.info()
train.isna().sum()
train.describe()
plt.subplots(figsize=(18,7))

plt.title("Répartition des outliers")

train.boxplot();
print(train.loc[train['trip_duration'] > 500000])
print(f"training set shape : {train.shape}")

train = train.loc[train['trip_duration']< 500000]

print(f"training set shape : {train.shape}")
corr = train.corr()

sns.heatmap(corr)
le = LabelEncoder()

le.fit(train['store_and_fwd_flag'])

train['store_and_fwd_flag'] = le.transform(train['store_and_fwd_flag'])

test['store_and_fwd_flag'] = le.transform(test['store_and_fwd_flag'])

train.head()
plt.subplots(figsize=(10,6))

plt.hist(train.trip_duration, bins=100)

plt.xlabel('trip_duration')

plt.ylabel('Frequency')

plt.show()
#Log transformation

plt.subplots(figsize=(10,6))

train['trip_duration'] = np.log1p(train['trip_duration'].values) 

plt.hist(train.trip_duration.values, bins=150)

plt.xlabel('log(trip_duration+1)')

plt.ylabel('Frequency')

plt.show()
train.columns
#### date features

## dates

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])



train['month'] = train['pickup_datetime'].dt.month

train['day'] = train['pickup_datetime'].dt.day

train['weekday'] = train['pickup_datetime'].dt.weekday

train['hour'] = train['pickup_datetime'].dt.hour

train['minute'] = train['pickup_datetime'].dt.minute



test['month'] = test['pickup_datetime'].dt.month

test['day'] = test['pickup_datetime'].dt.day

test['weekday'] = test['pickup_datetime'].dt.weekday

test['hour'] = test['pickup_datetime'].dt.hour

test['minute'] = test['pickup_datetime'].dt.minute



train.head()

train.columns
selection_train = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month", "day", "weekday", "hour", "minute"]

selection_test = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "month", "day", "weekday", "hour", "minute"]



#selection_train = ["passenger_count","month", "day", "weekday", "hour", "minute"]

#selection_test = ["passenger_count", "month", "day", "weekday", "hour", "minute"]



y_train = train["trip_duration"] # TARGET

X_train = train[selection_train] # FEATURES

X_test = test[selection_test]
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test) 
X_train.shape, X_test.shape, y_train.shape
y_train.head(5)
X_train.head(5)
X_test.head(5)
m1 = RandomForestRegressor()

#m1_scaled = RandomForestRegressor()
#print("Start cross validation ...")

#m1_scores = cross_val_score(m1, X_train, y_train, cv=5, scoring ="neg_mean_squared_log_error")

#m1_scores = cross_val_score(m1_scaled, X_train, y_train, cv=5, scoring ="neg_mean_squared_log_error")
#for i in range(len(m1_scores)):

#    m1_scores[i] = np.sqrt(abs(m1_scores[i]))

#print(m1_scores)

#pd.DataFrame(m1_scores).mean(), pd.DataFrame(m1_scores).std()

print("Start Training ...")

m1.fit(X_train, y_train)

#m1_scaled.fit(X_train_scaled, y_train)
y_pred = m1.predict(X_test)

#y_pred_scaled = m1_scaled.predict(X_test_scaled)
from sklearn.externals import joblib



# save the model to disk

filename = 'finalized_model.sav'

joblib.dump(m1, filename)



"""

# load the model from disk

loaded_model = joblib.load(filename)

result = loaded_model.score(X_test, Y_test)

print(result)

"""
!ls
#my_submission = pd.DataFrame({'id': test.id, 'trip_duration': y_pred})

#my_submission = pd.DataFrame({'id': test.id, 'trip_duration': y_pred_scaled})

my_submission = pd.DataFrame({'id': test.id, 'trip_duration': np.expm1(y_pred)})





my_submission.head()
my_submission.to_csv('submission.csv', index=False)

print("Submission done !")