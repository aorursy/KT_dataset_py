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
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv', parse_dates=["datetime"])

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv', parse_dates=["datetime"])



X = train.drop(['datetime','count','casual','registered'],1)

X['year'] = train['datetime'].dt.year

X['hour'] = train['datetime'].dt.hour



y = train['count']

test['year'] = test['datetime'].dt.year

test['hour'] = test['datetime'].dt.hour



test= test.drop(['datetime'],1)



X['season'] = X['season'].astype("category")

X['hour'] = X['hour'].astype("category")



test['season'] = test['season'].astype("category")

test['hour'] = test['hour'].astype("category")

from sklearn.ensemble import RandomForestRegressor as rf

model = rf(n_estimators=20)

model.fit(X, y)

model.score(X, y)

pred = model.predict(test)



submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

submission["count"] = pred

submission.to_csv("/kaggle/working/submission.csv", index=False)