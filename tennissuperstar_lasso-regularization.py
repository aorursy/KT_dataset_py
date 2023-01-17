# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



from pandas import DataFrame, Series

import matplotlib.pyplot as plt



import statsmodels.api as sm

from sklearn.cross_validation import train_test_split

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Train has the column 'SalePrice' and therefore it has 81 columns while test doesn't include

# SalePrice and therefore it only has 80 columns

# We are using the 79 variables (not including Id) in order to predict a value for SalePrice

# We are going to start by creating a linear regression on the numerical variables

# We will gauge how well our model is doing by first using train_test_split which will split 

# up our training data which has the SalePrice actual values

# Then we will run our data against the test set in order to produce the final submission



train.columns
# In order to train our classifier we need to split up the data into x and y

# X represents all the variables we are going to train with and y is the salePrice to predict

# We need to fill all na values with the mean of the column

# Before we can do this though we need to make sure that we only use numerical variables

# Therefore we should use the dummy function to replace all categorical with indicator 

# variables.



X = train.drop(['SalePrice'], axis=1)

X = pd.get_dummies(X)

Y = train['SalePrice']

X = X.fillna(X.mean())



# We want to split up the data into a train and test set

# Therefore we split up X into a trainX and a testX and the same for Y

# 1) We will train the model on trainX and trainY

# 2) We will predict values using testX

# 3) We will compare these values against testY



local_trainX, local_testX = train_test_split(X, test_size=0.2, random_state=123)

local_trainY, local_testY = train_test_split(Y, test_size=0.2, random_state=123)



clf = sm.OLS(local_trainY, local_trainX)

result = clf.fit()
# Now we have trained the model so the next step is to use the model in order to test on

# our test set

predictions = np.log(result.predict(local_testX) + 1)

local_testY = np.log(local_testY + 1)



# Mean squared error

def rmse (model):

    return np.log(np.sqrt(((predictions - local_testY) ** 2).mean())+1)
# Now let's look at the correlation matrix and remove the worse of any pairs of variables

# that have correlation > 0.8

X.corr()
# Going to try RandomForestRegressor



from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import train_test_split



X_train1, X_test1, y_train1, y_test1 = train_test_split(local_trainX, local_trainY)

clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)



clf.fit(X_train1, y_train1)

predictions = clf.predict(X_test1)



predictions = np.log(predictions + 1)

local_testY = np.log(local_testY + 1)

print(np.log(np.sqrt(((predictions - local_testY) ** 2).mean())+1))
# Regularization is important for regression models. Regularization

# smoothens out a regression model. This is important because

# some terms might be much larger than other terms.

# For example if you have coefficients that are ~1000 vs others 

# that are ~1 when you minimize these equations the terms 

# with large coefficients will effectively be reduced to 0.



# This code comes from the Regularized linear models script

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 70]

cv_ridge = [rmse(Ridge(alpha = alpha)).mean() for alpha in

           alphas]
# Now I want to create a graph that will tell me what

# value of alpha will result in the smallest rmse

cv_ridge = pd.Series(cv_ridge, index=alphas)

cv_ridge.plot()

plt.xlabel('alpha')

plt.ylabel('rmse')
# There seems to be a difference in how rmse is calculated

# however since we need to use the above defined function

# we will instead try out lasso regression.



from sklearn.cross_validation import train_test_split



X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train)

clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)



clf.fit(X_train1, y_train1)

y_pred = clf.predict(X_test1)
p = np.expm1(model_lasso.predict(local_trainX))

solution = pd.DataFrame({"id":test.Id, "SalePrice":p}, columns=['id', 'SalePrice'])

solution.to_csv("lasso_sol.csv", index = False)


