import os

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv")

test = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")
train.head()
# We will create one hot encoding for entry , exit direction fields for train, test set



dfen = pd.get_dummies(train["EntryHeading"],prefix = 'en')

dfex = pd.get_dummies(train["ExitHeading"],prefix = 'ex')

train = pd.concat([train,dfen],axis=1)

train = pd.concat([train,dfex],axis=1)
dfent = pd.get_dummies(test["EntryHeading"],prefix = 'en')

dfext = pd.get_dummies(test["ExitHeading"],prefix = 'ex')

test = pd.concat([test,dfent],axis=1)

test = pd.concat([test,dfext],axis=1)
train.shape,test.shape
train.columns
test.head()
train.columns
X = train[["IntersectionId","Hour","Weekend","Month",'en_E',

       'en_N', 'en_NE', 'en_NW', 'en_S', 'en_SE', 'en_SW', 'en_W', 'ex_E',

       'ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W']]

y1 = train["TotalTimeStopped_p20"]

y2 = train["TotalTimeStopped_p50"]

y3 = train["TotalTimeStopped_p80"]

y4 = train["DistanceToFirstStop_p20"]

y5 = train["DistanceToFirstStop_p50"]

y6 = train["DistanceToFirstStop_p80"]
testX = test[["IntersectionId","Hour","Weekend","Month",'en_E',

       'en_N', 'en_NE', 'en_NW', 'en_S', 'en_SE', 'en_SW', 'en_W', 'ex_E',

       'ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W']]
model = CatBoostRegressor(loss_function='RMSE')
model.fit(X, y1)

pred1 = model.predict(testX)

model.fit(X, y2)

pred2 = model.predict(testX)

model.fit(X, y3)

pred3 = model.predict(testX)

model.fit(X, y4)

pred4 = model.predict(testX)

model.fit(X, y5)

pred5 = model.predict(testX)

model.fit(X, y6)

pred6 = model.predict(testX)
# Appending all predictions

all_preds = []

for i in range(len(pred1)):

    for j in [pred1,pred2,pred3,pred4,pred5,pred6]:

        all_preds.append(j[i])
len(all_preds)
sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
sub["Target"] = all_preds
sub.to_csv("output.csv",index = False)