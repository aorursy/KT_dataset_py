import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# read the data

advertising = pd.read_csv("../input/advertising.csv")

advertising.head()
advertising.shape
advertising.info()
advertising.describe()
# visualise the data

sns.regplot( x='TV', y='Sales',data=advertising)
sns.regplot( x='Radio', y='Sales',data=advertising)
sns.regplot( x='Newspaper', y='Sales',data=advertising)
sns.pairplot( data=advertising, x_vars=['Newspaper','TV','Radio'],y_vars=['Sales'])
advertising.corr()
sns.heatmap(advertising.corr(), annot=True)
import statsmodels

import statsmodels.api as sm

import sklearn 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
# create x and y

X = advertising['TV']

y = advertising['Sales']
# train and test split

X_train ,X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=100)
X_train
# training the model 

X_train_sm = sm.add_constant(X_train)

X_train_sm.head()
# y = c + m1. X1

# y = c.const + m1.TV
# fitting the model

lr = sm.OLS(y_train, X_train_sm)#ols: ordinary least square

lr_model = lr.fit()

lr_model.params
# Sales = 6.94 + 0.05.TV

lr_model.summary()
#1. coef and p-value is very low

#2. R-squared is 81.6$, very high which is good 

#3. F-statistic is low => the fit is not by chance 
y_train_pred = lr_model.predict(X_train_sm)
plt.scatter(X_train, y_train)

plt.plot(X_train, 6.948 + 0.054*X_train,'r')

plt.show()
# error = f(y_train, y_train_pred)
res = y_train-y_train_pred
# plot the residual

plt.figure()

sns.distplot(res)

plt.title("Residual Plot")
# look for patterns in residual(we should not be able to identify)

plt.scatter(X_train, res)

plt.show()
# predictions on the test set (y_test_pred)

#evaluate the model, r-squared on the test
# add a constant

X_test_sm = sm.add_constant(X_test)

# pred on test

y_test_pred = lr_model.predict(X_test_sm)
#evaluate the model, r-squared on the test

# r-squared 

r2 = r2_score(y_true = y_test, y_pred =y_test_pred)

r2
# r2 on train

r2_score(y_true = y_train, y_pred = y_train_pred)
# mean squared error 

mean_squared_error( y_true = y_test, y_pred= y_test_pred)
plt.scatter(X_test, y_test)

plt.plot(X_test , y_test_pred ,'r')

plt.show()
# train test split

X_train ,X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=100)
#reshape to (40,1)

X_train_lm=X_train.values.reshape(-1,1)

X_test_lm=X_test.values.reshape(-1,1)

#(140, )

#(140,1)
# steps in sklearn model building



# 1. Create an object of linear regression

lm = LinearRegression()

# 2. Fit the model 

lm.fit(X_train_lm,y_train)
# 3. see the params, make predicitions(train,test)

print(lm.coef_)

print(lm.intercept_)
#make predictions

y_train_pred = lm.predict(X_train_lm)

y_test_pred = lm.predict(X_test_lm)
# 4.evaluate (r2,etc.)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

print(r2_score(y_true=y_test, y_pred=y_test_pred))