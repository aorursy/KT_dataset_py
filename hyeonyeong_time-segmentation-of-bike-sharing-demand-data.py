# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=["datetime"])
print(train.shape)

train.head()
print(test.shape)

test.head()
train["datetime-year"] = train["datetime"].dt.year

train["datetime-month"] = train["datetime"].dt.month

train["datetime-day"] = train["datetime"].dt.day

train["datetime-hour"] = train["datetime"].dt.hour

train["datetime-minute"] = train["datetime"].dt.minute

train["datetime-second"] = train["datetime"].dt.second

train["datetime-dayofweek"] = train["datetime"].dt.dayofweek

print(train.shape)

train[["datetime", "datetime-year", "datetime-month", "datetime-day", "datetime-hour", "datetime-minute", "datetime-second", "datetime-dayofweek"]].head()
train.loc[train["datetime-dayofweek"] == 0, "datetime-dayofweek(humanized)"] = "Monday"

train.loc[train["datetime-dayofweek"] == 1, "datetime-dayofweek(humanized)"] = "Tuesday"

train.loc[train["datetime-dayofweek"] == 2, "datetime-dayofweek(humanized)"] = "Wednesday"

train.loc[train["datetime-dayofweek"] == 3, "datetime-dayofweek(humanized)"] = "Thursday"

train.loc[train["datetime-dayofweek"] == 4, "datetime-dayofweek(humanized)"] = "Friday"

train.loc[train["datetime-dayofweek"] == 5, "datetime-dayofweek(humanized)"] = "Saturday"

train.loc[train["datetime-dayofweek"] == 6, "datetime-dayofweek(humanized)"] = "Sunday"

print(train.shape)

train[["datetime", "datetime-dayofweek", "datetime-dayofweek(humanized)"]].head()
test["datetime-year"] = test["datetime"].dt.year

test["datetime-month"] = test["datetime"].dt.month

test["datetime-day"] = test["datetime"].dt.day

test["datetime-hour"] = test["datetime"].dt.hour

test["datetime-minute"] = test["datetime"].dt.minute

test["datetime-second"] = test["datetime"].dt.second

test["datetime-dayofweek"] = test["datetime"].dt.dayofweek

print(test.shape)

test[["datetime", "datetime-year", "datetime-month", "datetime-day", "datetime-hour", "datetime-minute", "datetime-second", "datetime-dayofweek"]].head()
test.loc[test["datetime-dayofweek"] == 0, "datetime-dayofweek(humanized)"] = "Monday"

test.loc[test["datetime-dayofweek"] == 1, "datetime-dayofweek(humanized)"] = "Tuesday"

test.loc[test["datetime-dayofweek"] == 2, "datetime-dayofweek(humanized)"] = "Wednesday"

test.loc[test["datetime-dayofweek"] == 3, "datetime-dayofweek(humanized)"] = "Thursday"

test.loc[test["datetime-dayofweek"] == 4, "datetime-dayofweek(humanized)"] = "Friday"

test.loc[test["datetime-dayofweek"] == 5, "datetime-dayofweek(humanized)"] = "Saturday"

test.loc[test["datetime-dayofweek"] == 6, "datetime-dayofweek(humanized)"] = "Sunday"

print(test.shape)

test[["datetime", "datetime-dayofweek", "datetime-dayofweek(humanized)"]].head()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

figure.set_size_inches(18, 8)

sns.barplot(data=train, x="datetime-year", y="count", ax=ax1)

sns.barplot(data=train, x="datetime-month", y="count", ax=ax2)

sns.barplot(data=train, x="datetime-day", y="count", ax=ax3)

sns.barplot(data=train, x="datetime-hour", y="count", ax=ax4)

sns.barplot(data=train, x="datetime-minute", y="count", ax=ax5)

sns.barplot(data=train, x="datetime-second", y="count", ax=ax6)
train["datetime-year(str)"] = train["datetime-year"].astype('str')

train["datetime-month(str)"] = train["datetime-month"].astype('str')

train["datetime-year_month"] = train["datetime-year(str)"] + "-" + train["datetime-month(str)"]

print(train.shape)

train[["datetime", "datetime-year_month"]].head()
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

figure.set_size_inches(18, 4)

sns.barplot(data=train, x="datetime-year", y="count", ax=ax1)

sns.barplot(data=train, x="datetime-month", y="count", ax=ax2)

figure, ax3 = plt.subplots(nrows=1, ncols=1)

figure.set_size_inches(18, 4)

sns.barplot(data=train, x="datetime-year_month", y="count", ax=ax3)
figure, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

figure.set_size_inches(18, 12)

sns.pointplot(data=train, x="datetime-hour", y="count", ax=ax1)

sns.pointplot(data=train, x="datetime-hour", y="count", hue="workingday", ax=ax2)

sns.pointplot(data=train, x="datetime-hour", y="count", hue="datetime-dayofweek(humanized)", ax=ax3)
feature_names = ["datetime-year", "season", "datetime-hour", "datetime-dayofweek", "workingday", "holiday", 

                 "weather", "humidity", "temp", "atemp", "windspeed"]

feature_names
label_name = "count"

label_name
X_train = train[feature_names]

print(X_train.shape)

X_train.head()
X_test = test[feature_names]

print(X_test.shape)

X_test.head()
y_train = train[label_name]

print(y_train.shape)

y_train.head()
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,

                              random_state=37)

model
from sklearn.metrics import make_scorer

def rmsle(predict, actual):

    predict = np.array(predict)

    actual = np.array(actual)

    log_predict = np.log(predict + 1)

    log_actual = np.log(actual + 1)

    distance = log_predict - log_actual

    square_distance = distance ** 2

    mean_square_distance = square_distance.mean()

    score = np.sqrt(mean_square_distance)

    return score



rmsle_score = make_scorer(rmsle)

rmsle_score
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, X_train, y_train,

                        cv=20, scoring=rmsle_score).mean()

print("Score = {0:.5f}".format(score))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions.shape)

predictions
submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

print(submission.shape)

submission.head()
submission["count"] = predictions

print(submission.shape)

submission.head()
submission.to_csv("submission.csv", index=False)