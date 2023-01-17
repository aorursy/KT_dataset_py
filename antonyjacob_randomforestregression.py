import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

dataset = pd.read_csv('../input/dataregression/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, -1].values
from sklearn.ensemble import RandomForestRegressor

regg=RandomForestRegressor(n_estimators=300,random_state=0)

regg.fit(X,y)
y_pred = regg.predict([[6.5]])

print(y_pred)
X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regg.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (random forest regg Model)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()