import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
startups = pd.read_csv("../input/seaborn-50-startups-dataset/50_Startups.csv")
df = startups.copy() # yapılan değişiklikler startups değişkenini etkilemesin diye kopyasını oluşturalım
df.head()
df.info()
df.shape
df.isna().sum()
df.corr()
corr=df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df, color = "red")
df.hist(figsize =(13,8), color = "purple")
plt.show()
df.describe().T
df["State"].unique()
pd.get_dummies(df["State"])
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'State')
df.drop(["State"], axis = 1 , inplace = True) # state özniteliğini silelim
df = pd.concat([df, dfDummies], axis=1) #dummy olarak yaratılan State'leri ekleyelim
df.drop(["State_California"], axis = 1 , inplace = True) # State_California sütununu kaldıralım ve tekrar bakalım
df.head()
X = df.drop(["Profit"] , axis = 1) # bağımsız değişkenler
Y = df["Profit"] # bağımlı değişken Profit
Y
X
from sklearn.model_selection import train_test_split # train_test_split kullanmak için çekirdeğe dahil edelim
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 2/5, random_state = 20, shuffle=1)
# 50 veriden 20'sini test için kullanacağız
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
df = pd.DataFrame({"Gerçek Değer" : y_test, "Tahmin Edilen Değer" : y_pred})
df
from sklearn import metrics
print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test ,y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R Squared:", model.score(X_train, y_train))





