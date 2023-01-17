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
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Train_data = pd.read_csv("../input/train.csv")
Test_data = pd.read_csv("../input/test.csv")
ID = Test_data['Id']
Train_data.head()
Test_data.head()
train_levels = Train_data.loc[(Train_data['City'].notnull())]
City_counts = train_levels['City'].value_counts().sort_index().to_frame()
City_counts
train_levels = Train_data.loc[(Train_data['Type'].notnull())]
label_counts = train_levels['Type'].value_counts().sort_index().to_frame()
label_counts
# 将字符 lie 删除
del Train_data["Open Date"]
del Train_data["City"]
del Train_data["City Group"]
del Train_data["Type"]
del Train_data["Id"]

del Test_data["Open Date"]
del Test_data["City"]
del Test_data["City Group"]
del Test_data["Type"]
del Test_data["Id"]
#将空值数据用 0 填充
Train_data = Train_data.fillna(0)
Test_data = Test_data.fillna(0)
Test_data.head(10)
#Regression on everything
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import numpy

sns.set_context("notebook", font_scale=1.11)
sns.set_style("ticks")

yTrain = Train_data['revenue'].apply(numpy.log)
Train_data = Train_data.drop(["revenue"],1)
xTrain = pd.DataFrame(Train_data)
xTest = pd.DataFrame(Test_data)

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

cls = RandomForestRegressor(n_estimators=170)
cls.fit(xTrain, yTrain)
pred = cls.predict(xTest)
pred = numpy.exp(pred)
closs = cls.score(xTrain, yTrain)
closs
pred = cls.predict(xTest)
pred = numpy.exp(pred)
pred
read_test = {
    "Id":ID,
    "Prediction":pred
}
read_ = pd.DataFrame(read_test)
read_.to_csv("sample_submission.csv",index=False)
