import pandas as pd

import numpy as np

import pylab as py

import matplotlib.pyplot as plt

%matplotlib inline
my_data = pd.read_csv("../input/random-linear-regression/train.csv")

my_data =my_data.dropna()

my_data.head()
plt.scatter(my_data.x,my_data.y,edgecolor="red")

plt.title("Given Dataset",size= 20)

plt.xlabel("x",size= 20)

plt.ylabel("y",size= 20)

plt.show()
msk = np.random.rand(len(my_data)) < 0.3

train = my_data[msk]

test = my_data[~msk]
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['x']])

train_y = np.asanyarray(train[["y"]])

regr.fit(train_x,train_y)
print("The Coefficients: ",regr.coef_)

print("The Intercept is: ",regr.intercept_)
plt.scatter(train.x,train.y)

plt.plot(train_x,train_x*regr.coef_[0][0] + regr.intercept_[0],color ='red',alpha = 1,linewidth =2)

plt.legend(["Trained data","Orignal data"])

plt.title("Given Dataset after Training",size= 20)

plt.xlabel("x",size = 20)

plt.ylabel("y",size = 20)

plt.show()
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[["x"]])

test_y = np.asanyarray(test[["y"]])



y_hat = regr.predict(test_x)



print("The Mean Absolute Error(MAE) is: ",np.mean(np.absolute(y_hat - test_y)))

print("The Mean Squared Error(MSE) is: ",np.mean(np.absolute(y_hat - test_y)**2))

print("The R2 Score(R2) is: ",r2_score(y_hat,test_y))
test_data = pd.read_csv("../input/random-linear-regression/test.csv")

test_data.head()
test_data_x = test_data[["x"]]
test_data_y = test_data[["y"]]

pred = regr.predict(test_data_x)
print("The Mean Absolute Error(MAE) is: ",np.mean(np.absolute(pred - test_data_y)))

print("The Mean Squared Error(MSE) is: ",np.mean(np.absolute(pred - test_data_y)**2))

print("The R2 Score(R2) is: ",r2_score(pred,test_data_y))