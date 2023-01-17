# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df  = pd.read_csv('../input/shotputt_powerclean.csv')
df.head()
df.isnull().any()
X = df.iloc[:, 0:1].values

y = df.iloc[:, 1].values

plt.scatter(x = X,y =y,color = 'olive')
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'black')

plt.plot(X_grid, regressor.predict(X_grid), color = 'green')

plt.title('Relationship of Strength to Performance among Shot Putters (Decision Tree Regression)')

plt.xlabel('Power Clean')

plt.ylabel('Shotput')

plt.show()

accuracy = regressor.score(X_test,y_test)

print("With decsion tree we got --->",(accuracy*(100)).round(2),"Accuracy")
# Predicting a new result

y_pred = regressor.predict([[120]])

print(y_pred)