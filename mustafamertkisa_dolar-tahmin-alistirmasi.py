#kutuphaneleri yukleme



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
#veri yukleme



veriler = pd.read_csv("../input/usdtrycsv/usdtry.csv")
#data frame dilimleme



x = veriler.iloc[:,0:1]

y = veriler.iloc[:,1:]



#numpy array donusumu



X = x.values

Y = y.values



#verilerin dagilimi



plt.scatter(X,Y,color="orange")

plt.xlabel("Gün") 

plt.ylabel("Fiyat")

plt.title("Verilerin Dağılımı")

plt.show()
#lineer regresyon



linreg = LinearRegression()

linreg.fit(X,Y)



#lineer regresyon gorsellestirme



plt.scatter(X,Y,color="orange")

plt.plot(x,linreg.predict(X), color ="green")

plt.xlabel("Gün") 

plt.ylabel("Fiyat")

plt.title("Lineer Regresyon")

plt.show()



#lineer tahmin



print("Lineer Regresyon Tahmin:",linreg.predict([[280]]))
#polinomsal regresyon



polreg = PolynomialFeatures(degree=6)

xpol = polreg.fit_transform(X)

linreg = LinearRegression()

linreg.fit(xpol,y)



#polinomsal regresyon gorsellestirme



plt.scatter(X,Y,color ="orange")

plt.plot(X,linreg.predict(polreg.fit_transform(X)), color = "green")

plt.xlabel("Gün") 

plt.ylabel("Fiyat")

plt.title("Polinomsal Regresyon (Derece:6)")

plt.show()



#polinomsal tahmin



print("Polinomsal Regresyon Tahmin:",linreg.predict(polreg.fit_transform([[280]])))