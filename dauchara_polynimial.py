import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)
X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures (degree=4)

X_poly = poly_reg.fit_transform(X)
X

y
plt.scatter(X,y)

plt.plot(X,y)
poly_reg.fit(X_poly,y)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly, y)

regressor.coef_
X_poly
lin_reg2 = LinearRegression ()

lin_reg2.fit(X_poly,y)
lin_reg2_pred = lin_reg2.predict(poly_reg.fit_transform(X))
plt.scatter(X,y,color = 'red')

plt.plot(X, lin_reg2_pred, color = 'blue')

plt.title('PolynomialRegression')

plt.xlabel('Position')

plt.ylabel("Salary")
a = np.array(6.5)

a
a = a.reshape(-1,1)
lin_reg2.predict(poly_reg.fit_transform(a))