import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_Y = StandardScaler()

X = sc_X.fit_transform(X)

y = np.squeeze(sc_Y.fit_transform(y.reshape(-1, 1)))

# the feature scaling will be done to both X and Y and still Y will remain the Vector
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])

y_pred = sc_Y.inverse_transform(y_pred)

print(y_pred)
plt.scatter(X, y, color = 'red')

plt.plot(X, regressor.predict(X), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()