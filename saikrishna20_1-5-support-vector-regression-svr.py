import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # Level in the data
y = dataset.iloc[:, -1].values # salary in the data
print(X)
print(y)
y.shape
y = y.reshape(len(y),1)
y.shape
print(y)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # scalar object of X
sc_y = StandardScaler() # scalar object of y
X = sc_X.fit_transform(X) # scaled values of X
y = sc_y.fit_transform(y) # scaled values of y
print(X)
print(y)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Gaussian rbf kernal is used in here
regressor.fit(X, y)
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()