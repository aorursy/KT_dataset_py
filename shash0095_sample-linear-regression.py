import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
training_dataset = pd.read_csv('../input/random-linear-regression/train.csv')
test_dataset = pd.read_csv('../input/random-linear-regression/test.csv')
training_dataset = training_dataset.dropna()
test_dataset = test_dataset.dropna()
y_train = training_dataset.iloc[:,-1].values
X_train = training_dataset.iloc[:,:-1].values
y_test = test_dataset.iloc[:,-1].values
X_test = test_dataset.iloc[:,:-1].values
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('X vs Y (Training set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('X vs Y (Test set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# equation format: y=b0 + b1x
print("value of b0 = " + str(regressor.coef_))
print("value of b1 = " + str(regressor.intercept_))