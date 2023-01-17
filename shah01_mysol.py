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
train = pd.read_csv("../input/train.csv", parse_dates=["Dates"])

test = pd.read_csv("../input/test.csv", parse_dates=["Dates"], index_col="Id")
train.head()
test.info()
train[train["X"] < -122.6].head()
train[train["X"] > -122.2]
train[train["Y"] < 37.6]
train[train["Y"] > 38.0]
test[test["X"] < -122.6]
test[test["X"] > -122.2]
test[test["Y"] < 37.6]
test[test["Y"] > 38.0]
train.info()
train = train.drop(train[train["X"]==-120.5].index)
test = test.drop(test[test["X"]==-120.5].index)
test.info()
train.info()
sorted(train["DayOfWeek"].unique())
sorted(test["DayOfWeek"].unique())
train["Dates"].dt.year.min()
train["Dates"].dt.year.max()
test["Dates"].dt.year.min()
test["Dates"].dt.year.max()
sampleSub=pd.read_csv("../input/sampleSubmission.csv")

sampleSub.head()
train.duplicated().any()
train = train.drop_duplicates()
train.duplicated().any()
category1 = sorted(train["Category"].unique())

category1
category2 = sorted(sampleSub.columns[1:])

category2
category1 == category2
train.isnull().any()
test.isnull().any()
train.info()
test.info()
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb



le = LabelEncoder()



le2 = LabelEncoder()

category = le2.fit_transform(train["Category"])



def table_preprocessing(table, le):

    table["PdDistrict"] = le.fit_transform(table["PdDistrict"])

    table["DayOfWeek"] = le.fit_transform(table["DayOfWeek"])

    table["Block"] = table["Address"].str.contains("Block", case=False).apply(lambda x: 1 if x == True else 0)

    table["n_days"] = (table["Dates"] - table["Dates"].min()).apply(lambda x: x.days)

    table["Year"] = table["Dates"].dt.year

    table["Month"] = table["Dates"].dt.month

    table["Day"] = table["Dates"].dt.day

    table["Hour"] = table["Dates"].dt.hour

    table["Minute"] = table["Dates"].dt.minute

    table["n_X"] = table["X"] - table["X"].min()

    table["n_Y"] = table["Y"] - table["Y"].min()

    table["X+Y"] = table["X"] + table["Y"]

    table["X-Y"] = table["X"] - table["Y"]

    table["X*Y"] = table["X"] * table["Y"]

    table.drop(["Dates"], axis=1, inplace=True)

    table.drop(["Address"], axis=1, inplace=True)



table_preprocessing(train, le)

table_preprocessing(test, le)

train.drop(["Category", "Descript", "Resolution"], axis=1, inplace=True)
train.info()
test.info()
train_data = lgb.Dataset(train, label=category, categorical_feature=['PdDistrict'])



params = {'boosting':'gbdt',

          'objective':'multiclass',

          'num_class':39,

          'max_delta_step':0.9,

          'min_data_in_leaf': 15,

          'learning_rate': 0.4,

          'max_bin': 465,

          'num_leaves': 41

         }



bst = lgb.train(params, train_data, 100)



predictions = bst.predict(test)



submission = pd.DataFrame(predictions, columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)

submission.to_csv('MySolution.csv', index_label='Id')
train.info()
test.info()