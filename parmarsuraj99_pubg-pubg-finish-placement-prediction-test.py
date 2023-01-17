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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
feats = train.columns[train.columns!="winPlacePerc"]
final_feats = feats.drop(["Id", "groupId", "matchId"])
x_train = train[final_feats]
x_test = test[final_feats]
y_train = train["winPlacePerc"]
import matplotlib.pyplot as plt
train.describe()
from sklearn.model_selection import train_test_split
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_train, y_train, test_size=0.2)
train = train.drop(train[train["headshotKills"]>22].index, axis=0)

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_train_s, y_train_s)
linear_pred = linear_reg.predict(x_test_s)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train_s)
scaler.transform(x_train_s)
scaler.transform(x_test_s)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test_s, linear_pred)
linear_pred_s = linear_pred/linear_pred.max()
linear_ans=linear_reg.predict(x_test)
linear_ans.shape
linear_ans_s = linear_ans/linear_ans.max()
linear_ans_s.shape
result = pd.DataFrame({"Id": test["Id"], "winPlacePerc": linear_ans})
result.to_csv("linear_pubg.csv", index=False) 

result