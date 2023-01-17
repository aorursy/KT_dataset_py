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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print("Train shape : ", df_train.shape)

print("Test shape : ", df_test.shape)



df_train.head()
df_test.head()
df_train.nunique()
alldata = pd.concat([df_train, df_test], sort=False)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cata_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

alldata[cata_cols] = alldata[cata_cols].apply(le.fit_transform)
alldata.head()
df_train = alldata[:len(df_train)]

df_test = alldata[len(df_train):]
y = df_train['y']

df_train.drop(['y'], axis=1, inplace=True)

df_test.drop(['y'], axis=1, inplace=True)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print("Train shape : ", df_train.shape)

print("Test shape : ", df_test.shape)





alldata = pd.concat([df_train, df_test], sort=False)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cata_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

alldata[cata_cols] = alldata[cata_cols].apply(le.fit_transform)

df_train = alldata.iloc[:len(df_train), :]

df_test = alldata.iloc[len(df_train):, :]



y = df_train['y']

df_train.drop(['y'], axis=1, inplace=True)

df_test.drop(['y'], axis=1, inplace=True)
from catboost import CatBoostRegressor
model = CatBoostRegressor(random_seed=42, depth = 4)
df_train.head()
model.fit(df_train, y, cat_features=[1,2,3,4,5,6,7,8])
preds = model.predict(df_test)
submmision = pd.read_csv("../input/sample_submission.csv")

submmision["y"] = preds

submmision.to_csv("benz_catboost.csv", index=False)