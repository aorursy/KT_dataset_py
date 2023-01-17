import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
startups = pd.read_csv("../input/startsups/50_Startups.csv")

df=startups.copy()
df.head(5)
df.info()
df.shape
df.isna().sum() # is not applicable fonksiyonu ile eksik verileri kontrol ediyoruz
df.corr()
sns.heatmap(df.corr(),

           xticklabels=df.corr().columns.values,

           yticklabels=df.corr().columns.values)
sns.scatterplot(x="R&D Spend",y="Profit",data=df, color = "red")
df.hist(figsize = (15,15),bins=14,color = 'red')



plt.show()
df.describe()
df.State.unique()
df_State = pd.get_dummies(df["State"])
df_State.head()
df.drop('State', axis=1 , inplace =True)

df=pd.concat([df,df_State],axis=1)
df.head() #tabloda oluşan sondan üçüncü sütunlar df_State sütunlarıdır. 
Bagimsiz_degiskenler = df.drop(["Profit"], axis=1)

Bagimli_degiskenler = df["Profit"]
Bagimli_degiskenler
Bagimsiz_degiskenler
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(Bagimsiz_degiskenler,Bagimli_degiskenler, test_size = 2/5, random_state = 2, shuffle=1) 
X_train
X_test
Y_train
Y_test
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)
df = pd.DataFrame({"Gerçek Degerler" : Y_test, "Predict Degerler" : y_pred,"Residual":abs(y_pred-Y_test)})

df
import sklearn.metrics as metrics 
print("Ortalama Mutlak Hatası(MAE):", metrics.mean_absolute_error(Y_test,y_pred))

print("Ortalama Kare Hatası(MSE):", metrics.mean_squared_error(Y_test ,y_pred))

print("Kök Ortalama Kare Hatası (RMSE):", np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
model.score(X_train, Y_train)