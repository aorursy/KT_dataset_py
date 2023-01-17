# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

df  = pd.read_csv('../input/shotputt_powerclean.csv')
df.head()
df.isnull().any()
X = df.iloc[:, 0:1].values

y = df.iloc[:, 1].values
plt.scatter(x = X,y =y,color = 'orange')
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)


# Visualising the Random Forest Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'black')

plt.plot(X_grid, regressor.predict(X_grid), color = 'orange')

plt.title('Relationship of Strength to Performance among Shot Putters (Random Forest Regression)')

plt.xlabel('Power Clean')

plt.ylabel('Shotput')

plt.show()
accuracy = regressor.score(X_test,y_test)

print((accuracy*(100)).round(2))
# Predicting a new result

y_pred = regressor.predict([[120]])

print(y_pred)