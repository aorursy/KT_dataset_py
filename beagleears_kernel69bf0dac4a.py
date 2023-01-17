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

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt

import seaborn as sns



from math import sqrt
# Read the data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
# One-hot encoding of SaleCondition data

#train = pd.concat([pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')], axis=1)

# train.drop(['SaleCondition'], axis=1, inplace=True)

train = pd.concat([train, pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')], axis=1)

train.drop(['SaleCondition'], axis=1, inplace=True)
# pull data into target (y) and predictors (X)

train_y = train.SalePrice



predictor_cols = ['MSSubClass',

                  'LotArea', 

                  'OverallQual', 

                  'YearBuilt',

                  'YrSold',

                  'TotRmsAbvGrd', 

                  'SaleCondition_Abnorml',

                  'SaleCondition_AdjLand',

                 'SaleCondition_Alloca',

                 'SaleCondition_Family',

                 'SaleCondition_Normal',

                 'SaleCondition_Partial']



# Create training predictors data

train_X = train[predictor_cols]



svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])



svm_clf.fit(train_X,train_y)
# Read the test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# One-hot encoding on test data

test = pd.concat([test, pd.get_dummies(test['SaleCondition'], prefix='SaleCondition')], axis=1)

test.drop(['SaleCondition'], axis=1, inplace=True)



# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[predictor_cols]

# Use the model to make predictions

predicted_prices = svm_clf.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
#print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ada)))

#print("RMSE: ", sqrt(mean_squared_error(test, predicted_prices)))
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission-bkbates.csv', index=False)