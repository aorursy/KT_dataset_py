import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
startups = pd.read_csv("../input/50startups/50_Startups.csv", sep = ",")

df = startups.copy()
df.head()
df.info()
df.shape # 50 gozlem ve 5 öznitelik.
df.isnull().sum() 
df["Marketing Spend"].unique() #Null deger yok fakat 0 değerini almis öznitelik var. 

#Veriye goz atıp bir mantiksizlik olup olmadigina bakalım. 

sns.scatterplot(data=df, x = "Marketing Spend", y = "Profit")

#Gorunen o ki marketing spend ile profit dogru orantili. Dikkatli bakilirsa bir degerin marketing spend verisi 

#50000 civari iken profiti beklenmeyecek derecede az. Biraz daha arastirma yapalim.
df.corr()# Korelasyon matrisine de baktiğimizda profit ile marketing spend arasında pozitif bir etkilesim oldugunu goruyoruz.
df[(df["Marketing Spend"] == 0)] # marketing spend degeri 0 olan verilere bakilirsa uygunsuzluk oldugu asikar. 

# 19 indexli veri icin market harcamasi 0 iken bu kari elde etmek biraz zor gibi. Bunu duzeltelim.
df.drop([19,47,48], inplace=True)# 0 olan kolonlari siliyorum.
df[(df["Marketing Spend"] == 0)] # tekrar kontrol ettigimizde sorun cozulmus gibi duruyor.
# 0 kontrolunu diger kolonlar icin de yapalim.
df["R&D Spend"].unique() #Bunda da 0 bulduk. Ayni islemleri tekrarlayalım.
sns.scatterplot(data=df, x = "R&D Spend", y = "Profit") # scatterplota baktigimizda iki degisken arasinda 

#dogru oranti oldugunu goruyoruz.
df[(df["R&D Spend"] == 0)] 

# R&D Spend degerinin 0 oldugu 49 indexli yere bakalım. Bir sorun olabilir. Bunu da düzeltelim.
df.drop([49], inplace=True)# 0 olan kolonlari siliyorum.
df[(df["R&D Spend"] == 0)] # tekrar kontrol ettigimizde sorun cozulmus gibi duruyor.
df["Profit"].unique() # 0 yok.
df["Administration"].unique() # 0 yok.
corr = df.corr() 

corr #Profit ile r&d arasındaki korelasyona baktığımızda oldukça yüksek olduğunu söyleyebiliriz. Demek ki r&d çalışmaları

#kar oranını dogrudan arttırıyor. Aynı şekilde marketing spend ile profit arasında da oldukça olumlu bir ilişki olduğunu

#görüyoruz. Arasirmaya devam edelim.

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
sns.scatterplot(x="R&D Spend", y= "Profit", data = df)
sns.scatterplot(data=df, x = "Profit", y = "Marketing Spend", hue="R&D Spend", size="Administration")
df.describe().T
df["State"].unique()
df["State"] = pd.Categorical(df["State"])
dfDummies = pd.get_dummies(df["State"], prefix="State")
dfDummies
df = pd.concat([df,dfDummies], axis=1)
df = df.drop(["State"], axis=1)

df = df.drop(["State_California"], axis=1)

df.head()
X = df.drop("Profit", axis=1)
y = df["Profit"]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
model = lm.fit(X_train,y_train)
y_pred = model.predict(X_test)

y_pred
df["lineer_p"] = model.predict(X)

df
sns.lmplot(x = "Profit", y = "lineer_p", data = df); #Gorsellestirip baktigimda egitimin basarili oldugunu goruyorum.
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(df["Profit"], df["lineer_p"])

MAE
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(df["Profit"], df["lineer_p"])

MSE
import math

RMSE = math.sqrt(MSE)

RMSE
model.score(X,y)
import statsmodels.api as sm
stmodel = sm.OLS(y,X).fit()
stmodel.summary()