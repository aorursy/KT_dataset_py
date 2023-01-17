#kutuphaneleri yukleme



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#veri yukleme



veriler = pd.read_csv("../input/maaslarcsv/maaslar.csv")



#data frame dilimleme



x = veriler.iloc[:,1:2]

y = veriler.iloc[:,2:]



#numpy array donusumu



X = x.values

Y = y.values



#lineer regresyon (linear regression)

#dogrusal model olusturma



from sklearn.linear_model import LinearRegression

linreg1 = LinearRegression()

linreg1.fit(X,Y)



#polinomsal regresyon (Polynomial Regression)

#dogrusal olmayan (nonLinear) model olusturma



#ikinci dereceden polinom

from sklearn.preprocessing import PolynomialFeatures

polreg2 = PolynomialFeatures(degree=2)

xpol2 = polreg2.fit_transform(X)

linreg2 = LinearRegression()

linreg2.fit(xpol2,y)



#dördüncü dereceden polinom

polreg3 = PolynomialFeatures(degree=4)

xpol3 = polreg3.fit_transform(X)

linreg3 = LinearRegression()

linreg3.fit(xpol3,y)
#gorsellestirme



plt.scatter(X,Y,color='pink')

plt.plot(x,linreg1.predict(X), color = 'brown')

plt.xlabel("egitim seviyesi") 

plt.ylabel("maas")

plt.title("lineer")

plt.show()



plt.scatter(X,Y,color = 'green')

plt.plot(X,linreg2.predict(polreg2.fit_transform(X)), color = 'blue')

plt.xlabel("egitim seviyesi") 

plt.ylabel("maas")

plt.title("2. dereceden polinom")

plt.show()



plt.scatter(X,Y,color = 'orange')

plt.plot(X,linreg3.predict(polreg3.fit_transform(X)), color = 'purple')

plt.xlabel("egitim seviyesi") 

plt.ylabel("maas")

plt.title("4. dereceden polinom")

plt.show()
#tahminler

print("lineer")

print(linreg1.predict([[11]]))

print(linreg1.predict([[6.6]]))

print("------------------------")

print("2.dereceden polinom")

print(linreg2.predict(polreg2.fit_transform([[11]])))

print(linreg2.predict(polreg2.fit_transform([[6.6]])))

print("------------------------")

print("4.dereceden polinom")

print(linreg3.predict(polreg3.fit_transform([[11]])))

print(linreg3.predict(polreg3.fit_transform([[6.6]])))