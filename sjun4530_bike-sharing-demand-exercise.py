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
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])
test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])
train["datetime"].dt.hour
train["datetime"].dt.hour.unique()
train["datetime"].dt.hour.value_counts()
train["year"] = train["datetime"].dt.year
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
test["year"] = test["datetime"].dt.year
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek
train["datetime"].dt.month.value_counts()
train["season"].value_counts()
test.shape
train.shape
train["datetime"].dt.day.value_counts()
test["datetime"].dt.day.value_counts()
train.head()
test.head()
y_casual = np.log1p(train.casual)
y_registered = np.log1p(train.registered)
train.drop(["datetime", "windspeed", "casual", "registered", "count"], 1)
test.drop(["datetime", "windspeed"], 1)
from lightgbm import LGBMRegressor
model = LGBMRegressor()
train.head()
y_casual.head()
import numpy as np
import pandas as pd
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])
test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])

train["year"] = train["datetime"].dt.year
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek

y_casual = np.log1p(train.casual)
y_registered = np.log1p(train.registered)
#y_train = np.log1p(train["count"])

train.drop(["datetime", "windspeed", "casual", "registered", "count"], 1, inplace=True)
test.drop(["datetime", "windspeed", ], 1, inplace=True)
model.fit(train, y_casual)
preds1 = model.predict(test)
model.fit(train, y_registered)
preds2 = model.predict(test)
submission = pd.read_csv("../input/sampleSubmission.csv")
submission["count"] = np.expm1(preds1) + np.expm1(preds2)
submission.to_csv("result.csv", index=False)
train = pd.read_csv("../input/train.csv")
import seaborn as sns
import matplotlib.pylab as plt
fig,axes = plt.subplots(2,1, figsize=(20,12))
sns.distplot(train["count"],ax=axes[0])
sns.distplot(np.log(train["count"]),ax=axes[1])
