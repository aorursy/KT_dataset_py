import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings('ignore')

#print(os.listdir("../input"))
data = pd.read_csv('../input/Position_Salaries.csv')

x = data.iloc[:,1:2].values

y = data.iloc[:, 2:3].values

# since the dataset is very small no need to split it into train test

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

x = sc_x.fit_transform(x)

y = sc_y.fit_transform(y)
# Fit SVR to the data

from sklearn.svm import SVR



regressor = SVR(kernel='rbf')

regressor.fit(x,y)
# Predicting a new result

from numpy import array

y_pred = regressor.predict(array(6.5).reshape(-1, 1))

y_pred = sc_y.inverse_transform(y_pred)
# Visualising the SVR results

plt.scatter(x,y, color='red')

plt.plot(x,regressor.predict(x), color='blue')

plt.title('SVR')

plt.xlabel('Position')

plt.ylabel('Salary')

plt.show()