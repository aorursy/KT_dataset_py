import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
TRAIN_PATH = os.path.join("..", "input", "train.csv")

TEST_PATH = os.path.join("..", "input", "test.csv")



df_train = pd.read_csv(TRAIN_PATH,index_col=0)

df_test = pd.read_csv(TEST_PATH, index_col=0)
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_train["pickup_datetime"].head()
df_train["trip_duration"].hist(bins=100)
df_train[df_train.trip_duration < 10000].trip_duration.hist(bins=100)
# We check which partitions of the data to study

# I think we would only take data which correspond to 95% of the data ?



DURATION_MAX = 3500 # With this value we are studying 99% of the data



df_train[df_train.trip_duration < DURATION_MAX].trip_duration.hist(bins=100)



print("Count of values >", DURATION_MAX, ": ", df_train[df_train.trip_duration > DURATION_MAX].trip_duration.count())

print("Count of values <=", DURATION_MAX, ": ", df_train[df_train.trip_duration <= DURATION_MAX].trip_duration.count())

print("Lost data : ", df_train[df_train.trip_duration > DURATION_MAX].trip_duration.count() / df_train[df_train.trip_duration <= DURATION_MAX].trip_duration.count() * 100, "%")



df_train = df_train[df_train.trip_duration < DURATION_MAX]
df_train.isna().sum()
df_train.duplicated().sum()
df_train[df_train.duplicated()]
df_train["pickup_hour"] = pd.to_datetime(df_train.pickup_datetime).dt.hour

df_train["pickup_day_of_week"] = pd.to_datetime(df_train.pickup_datetime).dt.dayofweek
df_train.pickup_hour.hist(bins=47)
mean_trip_duration_by_hour = []



for i in df_train.pickup_hour.unique():

    mean_trip_duration_by_hour.append(np.mean(df_train[df_train.pickup_hour == i].trip_duration))

fig, ax = plt.subplots(figsize=(22,8))

ax.scatter(x=df_train[:1000].pickup_hour, y=df_train[:1000].trip_duration)

ax.bar(df_train.pickup_hour.unique(), mean_trip_duration_by_hour)

plt.show()
df_train.pickup_day_of_week.hist(bins=13)
mean_trip_duration_by_day = []



for i in df_train.pickup_day_of_week.unique():

    mean_trip_duration_by_day.append(np.mean(df_train[df_train.pickup_day_of_week == i].trip_duration))

fig, ax = plt.subplots(figsize=(22,8))

ax.scatter(x=df_train[:1000].pickup_day_of_week, y=df_train[:1000].trip_duration)

ax.bar(df_train.pickup_day_of_week.unique(), mean_trip_duration_by_day)

plt.show()
df_test["pickup_hour"] = pd.to_datetime(df_test.pickup_datetime).dt.hour

df_test["pickup_day_of_week"] = pd.to_datetime(df_test.pickup_datetime).dt.dayofweek
# 1st test using only basic columns

SELECTION = ["pickup_longitude", "dropoff_longitude", "pickup_latitude", "dropoff_latitude", "pickup_hour"]

TARGET = "trip_duration"
X_train = df_train[SELECTION]

y_train = df_train[TARGET]

X_test = df_test[SELECTION]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
# m1 = RandomForestRegressor(n_estimators=10)

# m1.fit(X_train, y_train)
m2 = RandomForestRegressor(n_estimators=15)

m2.fit(X_train, y_train)
# m3 = RandomForestRegressor(n_estimators=20)

# m3.fit(X_train, y_train)
m4 = RandomForestRegressor(n_estimators=15, min_samples_leaf=100, min_samples_split=150)

m4.fit(X_train, y_train)
cv_scores_1 = cross_val_score(m2, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

cv_scores_1
# cv_scores_2 = cross_val_score(m4, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

# cv_scores_2
# def rmse(test, pred):

#     return np.sqrt(mean_squared_error(test, pred))



def get_err(score):

    err_test = []

    for i in range(len(score)):

        err_test.append(np.sqrt(abs(score[i])))

    return err_test



print(np.mean(get_err(cv_scores_1)))

# print(np.mean(get_err(cv_scores_2)))
y_test_pred = m2.predict(X_test)

print(y_test_pred[:10])
d = { "id": df_test.index, "trip_duration": y_test_pred}

submission = pd.DataFrame(d)

submission.head()
submission.to_csv("submission.csv", index=0)