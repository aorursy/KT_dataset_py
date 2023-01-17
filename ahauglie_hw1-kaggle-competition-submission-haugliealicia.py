import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df.fillna(0)

df
import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as MSE

import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



# pull data into target (y) and predictors (X)

train_y = train.SalePrice

predictor_cols = ['LotArea', 'PoolArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']



# Create training predictors data

train_X = train[predictor_cols]



svm_reg = LinearSVR(epsilon=1.5)



svm_reg = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])



svm_reg.fit(train_X,train_y)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[predictor_cols]

# Use the model to make predictions

predicted_prices = svm_reg.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission-SVR_HauglieAlicia-4features.csv', index=False)