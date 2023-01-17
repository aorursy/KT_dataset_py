# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory

# print all the file/directories present at the path

import os

print(os.listdir("../input/"))
# importing the dataset

dataset = pd.read_csv('../input/Position_Salaries.csv')
dataset.head()
dataset.info()
dataset.isnull().sum()
plt.plot(dataset.iloc[:,1:-1],dataset.iloc[:,-1],color='red')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.title('Position Level VS Salary')

plt.show()
# matrix of features as X and dep variable as Y (convert dataframe to numpy array)

X = dataset.iloc[:,1:-1].values          #Level

Y = dataset.iloc[:,-1].values           #Salary
# Applying Desicion Tree Regressor



from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor()

reg.fit(X,Y)
# Predicting a new result

y_pred = reg.predict([[6.5]])
y_pred
X
# Visualising the Decision Tree Regression results (higher resolution)



X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')

plt.plot(X_grid, reg.predict(X_grid), color = 'blue')

plt.title('Decision Tree Regression')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()