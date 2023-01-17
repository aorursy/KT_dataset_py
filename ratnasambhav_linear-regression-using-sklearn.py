import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
training_set = pd.read_csv('../input/train.csv')
training_set = training_set.dropna()

test_set = pd.read_csv('../input/test.csv')
test_set = test_set.dropna()

x_train = training_set.as_matrix(['x'])
y_train = training_set.as_matrix(['y'])

x_test = test_set.as_matrix(['x'])
y_test = test_set.as_matrix(['y'])
plt.title('Traning Set')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x_train, y_train, color='red', marker='o', s=1)
plt.show()
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

print('R sq: ', lr.score(x_train, y_train))

print('Correlation: ', math.sqrt(lr.score(x_train, y_train)))

print("Coefficient for X: ", lr.coef_)

print("Intercept for X: ", lr.intercept_)

print("Regrssion line is: y = " + str(lr.intercept_[0]) + " + (x * " + str(lr.coef_[0][0]) + ")")
y_hat = lr.predict(x_test)

plt.scatter(x_test, y_test, color='green', marker='o', s=1, label='observed values')
plt.plot(x_test, y_hat, color='red', label='predicted values')
plt.legend()
plt.title('Comparision of observed and actuall values')
plt.xlabel('X')
plt.show()

plt.plot(y_hat, 'o', y_test, 'x')
plt.show()