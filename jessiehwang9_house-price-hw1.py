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
import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error
# Read the data(train)

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

columns_data=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

train_X=train[columns_data]

train_y=train.SalePrice
#SVM

svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])



svm_clf.fit(train_X,train_y)

#linear regression

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(train_X,train_y)
#polynominal

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_features.fit_transform(train_X)

lin_reg = LinearRegression()

lin_reg.fit(X_poly, train_y)
# Read the test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_X=test[columns_data]
#SVM

# Use the model to make predictions

predicted_prices = svm_clf.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
#linear regressor

predictions = lm.predict(test_X)

print(predictions)
#polynomial

X_test_poly = poly_features.fit_transform(test_X)

y_pred = lin_reg.predict(X_test_poly)

y=train_y.drop([1459])
from sklearn import metrics

print('SVM MSE:',metrics.mean_squared_error(y,predicted_prices))

print('Linear Regressor MSE:',metrics.mean_squared_error(y,predictions))

print('Polynomial MSE:',metrics.mean_squared_error(y,y_pred))
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})

my_submission.head()

my_submission.to_csv('Hw1_Predicted_Prices.csv',header=True, index=False)