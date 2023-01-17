# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#%% Importing the dataset

dataset = pd.read_csv('../input/Iris.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 5]
#%% Encoding categorical data, turn the species into simple numbers (0,1,2)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

#transforms to integer categories (words -> ints for example)

y = labelencoder_y.fit_transform(y)
#%% Splitting the dataset into the Training set and the Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
#%% Fitting the Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor as RFR

regressor = RFR(n_estimators = 300, random_state = 0)

regressor.fit(X_train, y_train)
#%% predict the results of the test set

y_pred = regressor.predict(X_test)

# round the results to nearest whole number

y_pred = np.ndarray.round(y_pred)
#Plot the actual values of the test set, these are the real values we set aside for testing

X_grid = np.arange(0, len(X_test), 1)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.title('Actual Observations')

plt.xlabel('Index Number')

plt.ylabel('Encoded Species')

plt.scatter(X_grid, y_test, color = 'red')
#Plot our predicted values for the same input as the actual results above

plt.scatter(X_grid, y_pred, color = 'blue')

plt.title('Predicted Species')

plt.xlabel('Index Number')

plt.ylabel('Encoded Species')
#Now lets Plot the expected values on top of the actual ones

plt.scatter(X_grid, y_test, color = 'red', s = 150, alpha = 0.5)

plt.scatter(X_grid, y_pred, color = 'blue', alpha = 0.5)

plt.title('Predicted Species over Actual Observations')

plt.xlabel('Index Number')

plt.ylabel('Encoded Species')