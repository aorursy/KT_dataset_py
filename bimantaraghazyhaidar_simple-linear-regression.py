
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("../input/random-linear-regression/train.csv")
# dataset =linear regression
dataset.shape
dataset.plot(x='y', y='x', style='o')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
X = dataset['x'].values.reshape(-1,1)
y = dataset['y'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()