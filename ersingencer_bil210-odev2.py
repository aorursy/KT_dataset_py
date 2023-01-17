import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
startups=pd.read_csv("../input/50-startup/50_Startups.csv");

df = startups.copy()
df.head(5)
df.info()
df.shape
df.isna().sum()
corr=df.corr()

corr
sns.heatmap(corr);
sns.scatterplot(x=df["R&D Spend"], y=df["Profit"]);
df.hist();
df.describe()
df.State.unique()
dummydf=pd.get_dummies(df.State)
newdf=df.drop(columns="State",axis=1)
dummydf.drop(columns="New York",axis=1,inplace=True);

dummydf
df=pd.concat([newdf,dummydf],axis=1)

df
X_bagimsiz = df[["Marketing Spend","R&D Spend"]]

y_bagimli = df[["Profit"]]
X_bagimsiz
y_bagimli
from sklearn.model_selection import train_test_split as testsplit

X_egitim,X_test,y_egitim,y_test = testsplit(X_bagimsiz,y_bagimli,test_size=0.2)
X_egitim
X_test
y_egitim
y_test;
from sklearn.linear_model import LinearRegression
lineerregresyon=LinearRegression()

model = lineerregresyon.fit(X_egitim,y_egitim)

model
y_pred = model.predict(X_test)

y_pred
df_y_test=pd.DataFrame(y_test);
df_y_predict = pd.DataFrame(y_pred,columns=["Model tahmin değerleri"],index=y_test.index);

pd.concat([df_y_test,df_y_predict],axis=1)
from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)
metrics.mean_absolute_error(y_test,y_pred)
MSQ=metrics.mean_squared_error(y_test,y_pred)

from math import sqrt

sqrt(MSQ)