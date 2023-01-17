#kutuphaneleri yukleme



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import r2_score

import statsmodels.api as sm



#veri yukleme



veriler = pd.read_csv("../input/maaslaryeni/maaslar_yeni.csv")



#data frame dilimleme



x = veriler.iloc[:,2:5]

y = veriler.iloc[:,5:]



#numpy array donusumu



X = x.values

Y = y.values



#korelasyon



print(veriler.corr())



#lineer regresyon (linear regression)

#dogrusal model olusturma



from sklearn.linear_model import LinearRegression

linreg1 = LinearRegression()

linreg1.fit(X,Y)



print("*******************************")

model1 = sm.OLS(linreg1.predict(X),X)

print(model1.fit().summary())



print("---------Linear R2 degeri---------")

print(r2_score(Y, linreg1.predict(X)))

print("*******************************")



#polinomsal regresyon (Polynomial Regression)

#dogrusal olmayan (nonLinear) model olusturma

#dördüncü dereceden polinom



from sklearn.preprocessing import PolynomialFeatures

polreg3 = PolynomialFeatures(degree=4)

xpol3 = polreg3.fit_transform(X)

linreg3 = LinearRegression()

linreg3.fit(xpol3,y)



print("*******************************")

model2 = sm.OLS(linreg3.predict(polreg3.fit_transform(X)),X)

print(model2.fit().summary())



print("---------Polynomial R2 degeri---------")

print(r2_score(Y, linreg3.predict(polreg3.fit_transform(X))))

print("*******************************")





#destek vektör regresyon kullanımı için verilerin olceklenmesi



from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()

y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))



#support vector regression



from sklearn.svm import SVR

svrreg = SVR(kernel='rbf')

svrreg.fit(x_olcekli,y_olcekli)



print("*******************************")

model3 = sm.OLS(svrreg.predict(x_olcekli),x_olcekli)

print(model3.fit().summary())



print("---------SVR R2 degeri---------")

print(r2_score(Y, svrreg.predict(x_olcekli)))

print("*******************************")



#karar ağacı regresyonu (decision tree regression)



from sklearn.tree import DecisionTreeRegressor

rdt = DecisionTreeRegressor(random_state=0)

rdt.fit(X,Y)



print("*******************************")

model4 = sm.OLS(rdt.predict(X),X)

print(model4.fit().summary())



print("---------Decision tree R2 degeri---------")

print(r2_score(Y, rdt.predict(X)))

print("*******************************")



#rassal agac regresyonu (random forest regression)



from sklearn.ensemble import RandomForestRegressor

rfreg=RandomForestRegressor(n_estimators = 10,random_state=0)

rfreg.fit(X,Y.ravel())



print("*******************************")

model4 = sm.OLS(rfreg.predict(X),X)

print(model4.fit().summary())



print("---------Random forest R2 degeri---------")

print(r2_score(Y, rfreg.predict(X)))

print("*******************************")