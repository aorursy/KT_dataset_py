import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('../input/linear-regression-dataset/Linear Regression - Sheet1.csv')
data.head()
fig = plt.figure()

ax = fig.add_subplot(1,1,1)



ax.plot(data['X'], data['Y'])

ax.set_xlabel('input - x')

ax.set_ylabel('target - y')

plt.show()
data.tail()
data = data.iloc[:298]
fig = plt.figure()

ax = fig.add_subplot(1,1,1)



ax.plot(data['X'], data['Y'])

ax.set_xlabel('input - x')

ax.set_ylabel('target - y')

plt.show()


X = data['X'].to_numpy().reshape(-1, 1)

y = data['Y'].to_numpy()

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X ,y , test_size=0.3, random_state = 2)
from sklearn.linear_model import LinearRegression



model = LinearRegression()
from sklearn.model_selection import cross_val_score



scores = cross_val_score(model, X_train, y_train, cv= 5)

print(scores)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score



MSE = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)



print('Mean squared error: ', MSE)

print('R2 Score: ', r2)
print('Slope :', model.coef_)

print('Intercept :',model.intercept_)



print('The line is of the form "y = ({:.3f}) * x + {:.3f}"'.format(model.coef_[0], model.intercept_))