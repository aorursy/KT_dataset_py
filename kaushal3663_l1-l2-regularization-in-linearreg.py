# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



#Import graphical plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Import Linear Regression Machine Learning Libraries

from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import r2_score
data = pd.read_csv('/kaggle/input/mpgCar.csv')

data.head()
X = data.drop(['mpg'], axis = 1) # independent variable

y = data[['mpg']] #dependent variable
X_s = preprocessing.scale(X)

X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe



y_s = preprocessing.scale(y)

y_s = pd.DataFrame(y_s, columns = y.columns)
X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)

X_train.shape
regression_model = LinearRegression()

regression_model.fit(X_train, y_train)



for idx, col_name in enumerate(X_train.columns):

    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))

    

intercept = regression_model.intercept_[0]

print('The intercept is {}'.format(intercept))


ridge_model = Ridge(alpha = 0.3)

ridge_model.fit(X_train, y_train)



print('Ridge model coef: {}'.format(ridge_model.coef_))


lasso_model = Lasso(alpha = 0.1)

lasso_model.fit(X_train, y_train)



print('Lasso model coef: {}'.format(lasso_model.coef_))
#Simple Linear Model

print(regression_model.score(X_train, y_train))

print(regression_model.score(X_test, y_test))



#Ridge

print(ridge_model.score(X_train, y_train))

print(ridge_model.score(X_test, y_test))

#Lasso

print(lasso_model.score(X_train, y_train))

print(lasso_model.score(X_test, y_test))
data_train_test = pd.concat([X_train, y_train], axis =1)

data_train_test.head()
import statsmodels.formula.api as smf

ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin', data = data_train_test).fit()

ols1.params
mse  = np.mean((regression_model.predict(X_test)-y_test)**2)
import math

rmse = math.sqrt(mse)

print('Root Mean Squared Error: {}'.format(rmse))


fig = plt.figure(figsize=(10,8))

sns.residplot(x= X_test['hp'], y= y_test['mpg'], color='green', lowess=True )





fig = plt.figure(figsize=(10,8))

sns.residplot(x= X_test['acc'], y= y_test['mpg'], color='green', lowess=True )
y_pred = regression_model.predict(X_test)
plt.scatter(y_test['mpg'], y_pred)