import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
startups = pd.read_csv("../input/50-startups/50_Startups.csv")
df = startups.copy()
df
df.head()
df.info()
df.shape
df.isnull().sum()
corr = df.corr()
corr
sns.heatmap(corr,
           xticklabels = corr.columns.values,
           yticklabels = corr.columns.values);
#1-R&D Spend ile Profit arasında pozitif güçlü bir ilişki vardır.
#2-Yine Marketing Spend ile Profit arasında ilki kadar güçlü olmasa da pozitif bir ilişki vardır.
#3-Yine R&D Spend ile Marketing Spend arasında ilki kadar güçlü olmasa da pozitif bir ilişki vardır.
sns.scatterplot(df["R&D Spend"],df["Profit"])
#sns.distplot(df["Profit"] ,bins=16,color ="purple");
#sns.distplot(df["R&D Spend"] ,bins=16,color ="purple");
#sns.distplot(df["Administration"] ,bins=16,color ="purple");
sns.distplot(df["Marketing Spend"] ,bins=16,color ="purple");
df.describe().T
df["State"].unique()
df_State = pd.get_dummies(df["State"])
df_State
df = pd.concat([df,df_State],axis =1)
df.head()


df.drop(["State","California"], axis = 1, inplace = True)
df
X = df.drop("Profit", axis = 1)
y = df["Profit"]
X #bağımsız değişkenler
y #bağımlı değişken(profit)
X_train =df["R&D Spend"].copy()
X_train2 =df["Marketing Spend"].copy()
X_test =df["Administration"].copy()
X_test2 =df["Florida"].copy()
X_test3 = df["New York"].copy()
y_test =df["Profit"].copy()
X_train
X_train2
y_test
X_test
from sklearn.linear_model import LinearRegression

linear_regresyon = LinearRegression()



model = linear_regresyon.fit(X, y)
y_pred = linear_regresyon.predict(X)
df["y_pred"] = y_pred.copy()
df
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
MSE = mean_squared_error(df["Profit"], df["y_pred"])
MSE

RMSE = math.sqrt(MSE)
RMSE
linear_regresyon.score(X,y)




