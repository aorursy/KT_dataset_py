import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_table("../input/startup-profit/Assignment_1.txt")
dataset
dataset.shape
X = dataset.iloc[:,:-2].values

Y = dataset.iloc[:,-1].values
X
Y
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 1/10)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
Data = np.array([28754.33 , 118546.05 , 172795.67])

Y_manualtest = regressor.predict(Data.reshape(1,-1))

Y_manualtest
df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})

df
plt.scatter(Y_test , Y_pred)

plt.xlabel("Actual Profit")

plt.ylabel("Predicted Profit")