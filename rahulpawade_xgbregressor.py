import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")
train.describe()
train.info()
train.isnull().sum()
train = train.dropna()
train.isnull().sum()
train.shape
train["matchType"].value_counts()
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
train["matchType"] = l.fit_transform(train["matchType"])
train.corrwith(train["winPlacePerc"])
train = train.drop(columns=["Id","groupId","matchId"],axis=1)

x_train = train.drop(columns="winPlacePerc",axis=1)

y_train = train["winPlacePerc"]
test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
test.info()
test.isnull().sum()
test.shape
test["matchType"] = l.fit_transform(test["matchType"])
test = test.drop(columns=["Id","groupId","matchId"],axis=1)
x_test = test

x_test.shape
x_train.shape
y_train.shape
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
dfs = pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")
f = {"Id":dfs["Id"],"winPlacePerc":y_pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)