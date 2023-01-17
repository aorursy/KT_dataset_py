import pandas as pd #pandas kütüphanesini dahil ettim.

import numpy as np #numpy kütüphanesini dahil ettim.

import seaborn as sns #seaborn kütüphanesini dahil ettim.

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt #matplotlib kütüphanesini dahil ettim.
df = pd.read_csv("../input/50-startups/50_Startups.csv").copy() 
df.head()
df.info() #Veri çerçevesi 5 sütundan oluşmaktadır.Bunlardan State'in datatype'ı Object,geriye kalanların datatype'ı float64 tür.
df.shape #Veri çerçevesi 50 gözlem 5 öznitelikten oluşmaktadır.
df.isna().sum() #Hiçbir öznitelikte eksik veri bulunmamaktadır.
df.corr() #Korelasyon 1'e yaklaştıkça aralarındaki ilişki güçlenir.
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df); # Aralarında güçlğ bir ilişki var.
sns.distplot(df["Profit"], bins=16, color="black");
sns.distplot(df["Marketing Spend"], bins=16, color="black");

sns.distplot(df["Administration"], bins=16, color="black");
sns.distplot(df["R&D Spend"], bins=16, color="black");
sns.pairplot(df, x_vars=['R&D Spend','Administration','Marketing Spend'], y_vars='Profit', size=7, aspect=0.7)
df.describe().T
df["State"].unique()
df['State'] = pd.Categorical(df['State'])

dfDummies = pd.get_dummies(df['State'])

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.drop(["State","New York" ], axis = 1, inplace = True)

df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 3512, shuffle=1)
X_train.head()
X_test.head()
y_test.head()
y_train.head()
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_a = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})

df_a
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(y_test, y_pred)

MSE
from sklearn.metrics import mean_absolute_error



MSA = mean_absolute_error(y_test, y_pred)

MSA
import math



RMSE = math.sqrt(MSE)

RMSE
model.score(X, y)
import statsmodels.api as stat

stmodel = stat.OLS(y, X).fit()

stmodel.summary()