import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, -1].values.reshape(-1, 1) ## had to add the reshape here to convert the array from 2d to 1d NOTE
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(random_state=0)

reg.fit(X, y)
# Predicting a new result

"""y_pred = reg.predict(sc_X.transform([[6.5]]))

y_pred = sc_y.inverse_transform(y_pred)"""

y_pred = reg.predict([[6.5]])

y_pred
# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, reg.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (Decision Tree Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()