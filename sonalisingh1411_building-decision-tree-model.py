# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv("../input/Position_Salaries.csv")
#Checking the dataset

dataset
a = dataset.iloc[:, 1].values

b = dataset.iloc[:, 2].values

plt.scatter(a,b,color='red',s=50)

plt.xlabel('Level')

plt.ylabel('Salary')

plt.title('Level vs salary')

plt.show()

#Separating the dependent and independent variables

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values

print(X)
print(y)
# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)



y_pred = regressor.predict(np.array(6.5).reshape(-1,1))

print(y_pred)


# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'black')

plt.plot(X_grid, regressor.predict(X_grid), color = 'red')

plt.title('Truth or Bluff (Decision Tree Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()