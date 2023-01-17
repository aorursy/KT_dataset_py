# SVR



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/sp2vers/shotputt_powerclean.csv')
df.isnull().any()
X = df.iloc[:, 0:1].values

y = df.iloc[:, 1:2].values
plt.scatter(X, y, color = 'plum')
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X = sc_X.fit_transform(X)

y = sc_y.fit_transform(y)
# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X,y)

#visualizing the result

X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'green')

plt.plot(X_grid, regressor.predict(X_grid), color = 'brown')

plt.title('Relationship of Strength to Performance among Shot Putters (Polynomial Regression)')

plt.xlabel('Power Clean')

plt.ylabel('Shotput')

plt.show()
accuracy = regressor.score(X,y)

print((accuracy*(100)).round(2))
# Predicting a new result

y_pred = regressor.predict([[120]])

y_pred = sc_y.inverse_transform(y_pred)

print(y_pred)