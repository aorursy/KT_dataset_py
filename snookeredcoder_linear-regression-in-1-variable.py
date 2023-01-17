import math
import pandas as pd
import numpy as np
from sklearn import linear_model
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
clean_train_data = train_data.dropna()
clean_test_data = test_data.dropna()
print("Size of Training Set before cleaning data is ", train_data.size)
print("Size of Training Set After Cleaning data is ", clean_train_data.size)
clean_train_data.head()
train = np.array(clean_train_data)
test = np.array(clean_test_data)
train
x_train = np.array(train[:,0].reshape(-1,1))
y_train = np.array(train[:,1].reshape(-1,1))
x_test = np.array(test[:,0].reshape(-1,1))
y_test = np.array(test[:,1].reshape(-1,1))
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
r_squared = linear.score(x_train,y_train)
print("R Squared for Train is ", r_squared)
print("Correlation for Train is ", math.sqrt(r_squared))
y_predicted = linear.predict(x_test)
r_squared_test = linear.score(x_test,y_test)
correlation_test = math.sqrt(r_squared_test)
print("R Squared for test is ", r_squared_test)
print("Correlation for Test is", correlation_test)