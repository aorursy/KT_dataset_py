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
train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
train
train.info()
train.loc[0,'datetime'].split('-')[0]
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])

test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])
train.dtypes
train.head()
train["year"] = train["datetime"].dt.year

train["hour"] = train["datetime"].dt.hour

train["dayofweek"] = train["datetime"].dt.dayofweek



test["year"] = test["datetime"].dt.year

test["hour"] = test["datetime"].dt.hour

test["dayofweek"] = test["datetime"].dt.dayofweek
y_train = train["count"]

y_train = np.log1p(y_train)
train.drop(["datetime", "windspeed", "casual", "registered", "count"], 1, inplace=True)

test.drop(["datetime", "windspeed"], 1, inplace=True)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=100)

rf.fit(train,y_train)

preds = rf.predict(test)
submission = pd.read_csv('../input/sampleSubmission.csv')

submission.head()
submission["count"] = np.expm1(preds)

submission.head()
submission.to_csv("allrf.csv", index=False)