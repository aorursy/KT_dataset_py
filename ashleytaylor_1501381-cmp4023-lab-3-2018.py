# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#random number generator, setting the seed to 1
randomSet = np.random.RandomState(1)
#defining x with the random generator
x = 10 * randomSet.rand(50)
#defining y relating to x 
y = 6 * x + randomSet.rand(50)
#storing the values in x and y in the df, randomdf
randomdf = pd.DataFrame({'x': x,
'y': y}
 )
randomdf

#finding the correlation between x and y
#from the table show, it can be seen that y and x are closely correlate as the correlation is close to 1
randomdf.corr()
#scatter plot showing the relationship between x and y(correlation)
plt.scatter(x,y, color = 'purple')
#withholding 30% of the data to be test data
X_train, X_test, Y_train, Y_test = train_test_split(randomdf.x, randomdf.y, test_size = 0.30)
#if testing was done on all the data then it would cause overfitting(a model that would just repeat the
#labels of the samples that it has just seen would have a perfect score but would fail to predict
#anything useful on yet-unseen data.)
#creating a linear regression object
linreg = linear_model.LinearRegression()
#training the model base on the test data. In other words, using the test data to determine the pattern
#between the variables x and y
linreg.fit(pd.DataFrame(X_train), Y_train)
#after training the regression model with the train data, the regression model with the test data 
#for x was used to predict the values of y
test_predict = linreg.predict(pd.DataFrame(X_test))
test_predict
sns.residplot(X_train,Y_train, lowess=True, color="g")
#R^2 is atatisitical measure of how close the data are fitted to the regression line.
#This means that a percentage of the response variable, y, is explained by the independent variable x
linreg.score(pd.DataFrame(X_test), Y_test)
#the impact that x has on  x changes y changes as well. 
#when x is increase by 1, y will increase 6.00607404 
linreg.coef_
#the regression formula comes in the form af y = mx + c, therefore finding the 
#coefficient and intercept are necessary to deduce the formula
linreg.intercept_
#the regression formula takes the form of y = mx + c 
# y = 6.01309565x + 0.45885944936149414
# using the equation y = 6.01309565x + 0.45885944936149414, when x is 350, y = 2105.042336949362
6.01309565 * 350 + 0.45885944936149414
#variance shows how far the each data in the set is from the mean
r2_score(Y_test,test_predict)
#Mean Absolute Error
mean_absolute_error(Y_test,test_predict)
#Root Mean Squared Error
np.sqrt(mean_squared_error(Y_test, test_predict))
#Mean Squared Error
mean_squared_error(Y_test, test_predict)
