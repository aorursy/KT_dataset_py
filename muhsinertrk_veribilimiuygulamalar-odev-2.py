import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn import metrics

#Gerekli olan kütüphaneler.
startups = pd.read_csv("../input/startups/50_Startups.csv")

df=startups.copy()
df.head()
df.info()
df.shape

#Datasetimiz 50 gözlem ve 5 değişkenden oluşmaktadır.
df.isna().sum()#Datasetimizde eksik değişken bulunmuyor.Temiz bir veri.
df.corr()#Aralarında en güçlü ilişki R&D Spend ile Profit
corr = df.corr()

sns.heatmap(corr,

           xticklabels = corr.columns.values,

           yticklabels = corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df,color="purple");
df.hist(figsize = (15,15),color="purple")



plt.show()
df.describe().T

df["State"].unique()
df_state = pd.get_dummies(df["State"])
df_state.head()
df_state.columns = ['New York', 'California', 'Florida']
df_state.head()
df.drop(["State"], axis=1 , inplace =True)

df=pd.concat([df,df_state],axis=1)
df.drop(["Florida"], axis=1, inplace = True)
df.head()
X = df.drop("Profit", axis = 1)

Y = df["Profit"]
X# bağımsız değişkenler
Y# bağımlı değişken
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 2, shuffle=1)
X_train
X_test
Y_train
Y_test
model=LinearRegression()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)
df = pd.DataFrame({"Gerçek Değerler" : Y_test, "Tahmin Edilen" : y_pred,"Aradaki Fark":abs(y_pred-Y_test)})



df
print("MAE:", metrics.mean_absolute_error(Y_test,y_pred))

print("MSE:", metrics.mean_squared_error(Y_test ,y_pred))

print("RMSE:", np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

df.describe()["Aradaki Fark"]
print("R Squared:", model.score(X_train, Y_train))#R Squared değeri bire yakındır model başarılıdır.