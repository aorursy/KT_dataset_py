# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
class SimpleLinearRegression:
    coef = 0
    intercept = 0
    rsquared = 0
    def fit(self, x_train, y_train):
        sum_of_x = sum(x_train)
        sum_of_y = sum(y_train)
        sum_of_x2 = np.sum(np.square(x_train))
        sum_of_y2 = np.sum(np.square(y_train))
        dotproduct = np.dot(x_train,y_train)
        length = len(x_train)
        dif_x = sum_of_x2 - sum_of_x * sum_of_x/length
        dif_y = sum_of_y2 - sum_of_y * sum_of_y/length
        numerator = length * dotproduct - sum_of_x * sum_of_y
        denom = (length * sum_of_x2 - sum_of_x * sum_of_x) * (length * sum_of_y2 - (sum_of_y * sum_of_y))
        co = dotproduct - sum_of_x * sum_of_y / length
        self.rsquared = np.square(numerator / np.sqrt(denom))
        self.intercept = sum_of_y / length - ((co / dif_x) * sum_of_x/length)
        self.coef = co / dif_x
    def predict(self,x_test):
        return x_test * self.coef + self.intercept
        
x_train = np.array([ 1, 2, 3, 4])
y_train = np.array([ 2, 3, 4, 4])
slr = SimpleLinearRegression()
slr.fit(x_train,y_train)
print("Coefficient:", slr.coef)
print('Y-Intercept:',slr.intercept)
print('R-Squared:',slr.rsquared)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
print(lr.coef_)
print(lr.intercept_)