import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
d_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
d_train.info()
for i in d_train.columns:

   if (d_train[i].isnull().sum())<1400:

    d_train[i] = d_train[i]

   
d_train.info()
d_train_object = d_train.select_dtypes(include=object)
for i in d_train_object.columns:

 if d_train_object[i].count()!=1460:

    d_train_object[i] = d_train_object.fillna(d_train_object[i].mode()[0])

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()

for i in d_train_object.columns:

    d_train_object[i] = e.fit_transform(d_train_object[i])
d_train_num = d_train.select_dtypes(include=np.number)
for i in d_train_num.columns:

 if d_train_num[i].count()!=1460:

    d_train_num[i] = d_train_object.fillna(d_train_num[i].mean())
df_train = pd.concat([d_train_object,d_train_num],axis=1)
df_train = df_train.drop(columns="Id",axis=1)

df_train.shape
dict((df_train.corrwith(d_train["SalePrice"])))
d_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
d_test.shape
d_test_object = d_test.select_dtypes(include=object)

d_test_num = d_test.select_dtypes(include=np.number)
for i in d_test_object.columns:

 if d_test_object[i].count()!=1459:

    d_test_object[i] = d_test_object.fillna(d_test_object[i].mode()[0])

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()

for i in d_test_object.columns:

    d_test_object[i] = e.fit_transform(d_test_object[i])
for i in d_test_num.columns:

 if d_test_num[i].count()!=1459:

    d_test_num[i] = d_test_num.fillna(d_test_num[i].mean())
df_test = pd.concat([d_test_object,d_test_num],axis=1)
df_test.info()
df_test = df_test.drop(columns="Id",axis=1)
df_test.shape
x_train = df_train.drop(columns=["SalePrice","GarageYrBlt"],axis=1)

x_train.shape
y_train = df_train["SalePrice"]

y_train.shape
x_test = df_test.drop(columns="GarageYrBlt",axis=1)

x_test.shape
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
dfs = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
f = {"Id":dfs["Id"],"SalePrice":y_pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)

f.head()