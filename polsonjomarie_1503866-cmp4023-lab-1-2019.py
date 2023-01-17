# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.

#Question 1

X1 = np.random.randint(low=500, high=2000,size= 50)

X2 =  np.random.randint(low=100, high=500,size= 50)

X3 = X1 * 3 + np.random.rand()

Y = X3 +X2

df1 = pd.DataFrame(list(zip(X1,X2,X3,Y)),columns=['X1','X2','X3','Y'])

df1

#Question 2 - parsons correlation

df1.corr()

#Question 3 - graph shows that they are dependent on each other based on the linear shape

plt.scatter(X1,Y)
plt.scatter(X2,Y)
#Question 4

from sklearn.model_selection import train_test_split 

from sklearn import linear_model

import statsmodels.api as sm


X = df1[['X1','X2']]

Y = df1['Y']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)# used for smaller data

pd.DataFrame(X_test)
Reg = linear_model.LinearRegression()#It is used to determine the extent to which there is a linear relationship between a dependent variable and one or more independent variables.

model = Reg.fit(X_train, y_train)# train data become new data

predictions = Reg.predict(X_test)# predict future y values

plt.scatter(y_test, predictions)# 

print("R-squared:",model.score(X_test,y_test))

print('Intercept:', model.intercept_)

print('Coefficient:', model.coef_) 
#Question 5

import seaborn as sns

sns.residplot(y_test,predictions, lowess=True, color="r")# underfitted
#Question 6

r_sq = model.score(X_test,y_test)

print('coefficient of determination:', r_sq)

print('Intercept:', model.intercept_)

print('Coefficient:', model.coef_)

#The distance between the regression line's y values, and the data's y values is the error, then we square that.

#The line's squared error is either a mean or a sum of this, we'll simply sum it.
#Question 7



sns.pairplot(df1,x_vars=['X1','X2'],y_vars=['Y'],height=4)

# X1 relationship has a strong correlation
#Question 9

Reg = linear_model.LinearRegression()

model = Reg.fit(X_train, y_train)

predictions = Reg.predict(X_test)

print(predictions)
#Question 10

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

print('Variance score: %.2f' % r2_score(y_test, predictions))

print("Mean squared error: %.2f" % mean_squared_error(y_test,predictions))

print("Mean absolute error: %.2f" % mean_absolute_error(y_test,predictions))

print("Root Mean squared error: %.2f" %np.sqrt(mean_squared_error(y_test,predictions)))