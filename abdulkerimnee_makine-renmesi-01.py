import pandas as pd
df = pd.read_excel("../input/real-estate-valuation-dataset/Real estate valuation data set.xlsx")
df.head()
df=df.rename(columns={"X1 transaction date":"İşlem Tarihi",
                  "X2 house age":"Ev Yaşı",
                  "X3 distance to the nearest MRT station":"Metroya Olan Uzaklık",
                  "X4 number of convenience stores":"Civardaki Market Sayısı",
                  "X5 latitude":"Enlem",
                  "X6 longitude":"Boylam",
                  "Y house price of unit area":"Birim Alan Fiyatı"})
df.head()
X = df[["Metroya Olan Uzaklık"]]#bağımsız değişken
y=df[["Birim Alan Fiyatı"]]#bağımlı değişken
X.head()
y.head()
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X,y)#modeli kurmak
model.intercept_#(b0)
model.coef_#(b1)
45.85142706+(-0.00726205)*X[0:5]
model.predict(X)[0:5]
model.predict([[5],[50],[500]])
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.regplot(df["Metroya Olan Uzaklık"], df["Birim Alan Fiyatı"], ci=None, scatter_kws={'color':'r', 's':9})
plt.ylim(bottom=0);
gercek_y = y[0:10]
tahmin_edilen_y = pd.DataFrame(model.predict(X)[0:10])#numpy array'inden dataframe'ye çevirelim 
hatalar = pd.concat([gercek_y, tahmin_edilen_y], axis = 1)#dataframe'leri birleştirme
hatalar.columns = ["gercek_y","tahmin_edilen_y"]#Sütunlara isim verme
hatalar
hatalar["hata"] = hatalar["gercek_y"] - hatalar["tahmin_edilen_y"]
hatalar["hata_kareler"] = hatalar["hata"]**2
hatalar
import numpy as np
np.mean(hatalar["hata_kareler"])
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(gercek_y,tahmin_edilen_y)
MSE
RMSE = np.sqrt(MSE)
RMSE