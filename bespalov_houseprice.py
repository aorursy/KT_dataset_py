import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split


sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head()
train.isnull().sum()[train.isnull().sum()>0]
train = train.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu'], axis=1)

test = test.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu'], axis=1)
train.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)


regressor = RandomForestRegressor(n_estimators=300, random_state=0)

regressor.fit(X_train, y_train)



# Score model

regressor.score(X_train, y_train)