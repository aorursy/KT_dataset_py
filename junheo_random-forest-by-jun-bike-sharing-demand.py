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
import pandas as pd



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])

test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])
train.info()
train["year"] = train["datetime"].dt.year

train["hour"] = train["datetime"].dt.hour

train["dayofweek"] = train["datetime"].dt.dayofweek



test["year"] = test["datetime"].dt.year

test["hour"] = test["datetime"].dt.hour

test["dayofweek"] = test["datetime"].dt.dayofweek
train.head()
y_train = train["count"]
import numpy as np

y_train = np.log1p(y_train)
test.head()
train.drop(["datetime", "windspeed", "casual", "registered", "count"], 1, inplace=True)
train.head()
test.head()
test.drop(["datetime", "windspeed"], 1, inplace=True)
test.head()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(train,y_train)
preds = rf.predict(test)
submission=pd.read_csv("../input/sampleSubmission.csv")
submission["count"] = np.expm1(preds)
submission.to_csv("allrf.csv", index=False)