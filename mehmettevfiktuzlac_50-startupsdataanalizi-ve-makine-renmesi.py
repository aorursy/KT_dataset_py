import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
startups = pd.read_csv("../input/sp-startup/50_Startups.csv", sep=",")

df = startups.copy()
df.head() #baştan ilk beş gözlemimiz
df.info() 
df.shape
df.isna().sum() # görmüş olduğumuz gibi eksik verimiz bulunmamakta
corr = df.corr()

corr 
sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df); 
df.info() #info ile verileri tekrardan çağırıp sayısal olanları displot kullanarak histogram grafiğine dönüştürelim. 
sns.distplot(df["R&D Spend"], bins=16, color="purple");
sns.distplot(df["Administration"], bins=16, color="blue");
sns.distplot(df["Marketing Spend"], bins=16, color="gold");
sns.distplot(df["Profit"], bins=16, color="green");
df["State"].unique()
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'StateOf')
dfDummies.head() # tüm değerleri gösterip kalabalık yapmak istemedim.
df = pd.concat([df, dfDummies], axis = 1)

df.head()
df.drop(["State", "StateOf_New York"], axis = 1, inplace = True)
df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X.head(10)
y.head(10)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 # test size'ı %20 yaptım yani 10 veriye tekabül ediyor %80'ini de train yani eğitim için kullanıyorum 40 veriyi.
X_train.head(10)
X_test # görüldüğü gibi diğerlerinde baştan 10 tane getir demiştim bunda dememe gerek yok çünkü test %20 alıyor
y_train.head(10)
y_test # aynı şeklilde test olduğu için bu da 10 değer alacaktır.
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = model.predict([[100000, 85000, 200000, 0, 0]])

y_pred 
df["kar_tahmin"] = model.predict(X)

df.head(10)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(df["Profit"], df["kar_tahmin"])

MSE
import math

RMSE = math.sqrt(MSE)

RMSE
from sklearn.metrics import mean_absolute_error



MAE = mean_absolute_error(df["Profit"], df["kar_tahmin"])

MAE
model.score(X_train,y_train)
import statsmodels.api as sm

stmodel = sm.OLS(y, X).fit()
stmodel.summary()
df.drop(["StateOf_California", "StateOf_Florida", "kar_tahmin"], axis = 1, inplace = True)
df.head()
X1 = df.drop("Profit", axis = 1)

X1.head() # X'e yeni tablomuzu aktarıyorum ki eski tablo ile eğitmeyelim di mi?
y.head() # y'de bir değişiklik yapmadık zaten.
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0) 
lm = LinearRegression()

model2 = lm.fit(X1_train, y_train)
y_pred2 = model2.predict([[100000, 85000, 200000]])

y_pred2
df["kar_tahmin"] = model2.predict(X1)

df.head(10)
MSE1 = mean_squared_error(df["Profit"], df["kar_tahmin"])

MSE1
RMSE1 = math.sqrt(MSE1)

RMSE1 # 9053.031218703098 eski değeri
model2.score(X1_train,y_train) # r squared değerinin ise çok küçük bir düşüş yaşadığını görüyoruz.