import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import seaborn as sns
df = pd.read_csv('../input/slump-test/slump_test.data', sep=',')

df = df.drop(['No'], axis=1)
df.head()
def rmse(targets, predictions):

    return np.sqrt(((predictions - targets) ** 2).mean())
df.corr()
x = df.drop(['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)'], axis=1)

y = df['Compressive Strength (28-day)(Mpa)']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/5, random_state = 0)
x_train = X_train['Fly ash']

x_test = X_test['Fly ash']
x_train = np.expand_dims(x_train, axis=1)

x_test = np.expand_dims(x_test, axis=1)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(x_train, y_train)
reg.coef_
y_pred = reg.predict(x_test)
print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The RMSE score is {rmse(y_test, y_pred)}')
plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, reg.predict(x_train), color = 'blue')

plt.title('Fly ash vs Compressive Strength (Training set)')

plt.xlabel('Fly ash (X)')

plt.ylabel('Compressive Strength(Y)')



plt.show()
plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_test, reg.predict(x_test), color = 'blue')

plt.title('Fly ash vs Compressive Strength (Test set)')

plt.xlabel('Fly ash (X)')

plt.ylabel('Compressive Strength(Y)')

plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The RMSE score is {rmse(y_test, y_pred)}')
reg.coef_
plt.scatter(y_train, reg.predict(X_train))

plt.title('Predictions vs True Values (Training)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
plt.scatter(y_test, reg.predict(X_test))

plt.title('Predictions vs True Values (Test)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
df = pd.read_csv('../input/compensation/compensation.csv')
x = df['Exp']

y = df['Comp']
X = np.expand_dims(x, axis = 1)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)
lin_reg_2.coef_
print(f'The R2 score is: {r2_score(y, lin_reg_2.predict(X_poly))}')

print(f'The RMSE score is {rmse(y, lin_reg_2.predict(X_poly))}')
plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.title('Compensation vs Years of experience (Linear Regression)')

plt.xlabel('Years of experience')

plt.ylabel('Compensation')

plt.show()

plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Compensation vs Years of experience (Polynomial Regression)')

plt.xlabel('Years of experience')

plt.ylabel('Compensation')

plt.savefig('test.jpg')

plt.show()

from sklearn.linear_model import Ridge

reg = Ridge()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The RMSE score is {rmse(y_test, y_pred)}')
reg.coef_
plt.scatter(y_train, reg.predict(X_train))

plt.title('Predictions vs True Values (Training)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
plt.scatter(y_test, reg.predict(X_test))

plt.title('Predictions vs True Values (Test)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
from sklearn.linear_model import Lasso

reg = Lasso(alpha=1.2)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The RMSE score is {rmse(y_test, y_pred)}')
reg.coef_
plt.scatter(y_train, reg.predict(X_train))

plt.title('Predictions vs True Values (Training)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
plt.scatter(y_test, reg.predict(X_test))

plt.title('Predictions vs True Values (Test)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
from sklearn.linear_model import ElasticNet

reg = ElasticNet()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The RMSE score is {rmse(y_test, y_pred)}')
reg.coef_
plt.scatter(y_train, reg.predict(X_train))

plt.title('Predictions vs True Values (Training)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
plt.scatter(y_test, reg.predict(X_test))

plt.title('Predictions vs True Values (Test)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
from sklearn.linear_model import BayesianRidge

reg = BayesianRidge()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The RMSE score is {rmse(y_test, y_pred)}')
reg.coef_
plt.scatter(y_train, reg.predict(X_train))

plt.title('Predictions vs True Values (Training)')

plt.xlabel('True Values')

plt.ylabel('Prediction')
plt.scatter(y_test, reg.predict(X_test))

plt.title('Predictions vs True Values (Test)')

plt.xlabel('True Values')

plt.ylabel('Prediction')