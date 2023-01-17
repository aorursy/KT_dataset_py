import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Train data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#Train

train_y = train.SalePrice

predictor_cols = ['GrLivArea', 'OverallQual','1stFlrSF']

train_X = train[predictor_cols]

svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])

svm_clf.fit(train_X,train_y)
#Test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Test

test_X = test[predictor_cols]

predicted_prices = svm_clf.predict(test_X)

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission-SVR_AllegraFrickel.csv', index=False)