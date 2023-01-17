# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.describe()
train.head()
train.tail()
train.shape , test.shape

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns

msno.bar(train.sample(100))

my_imputer = SimpleImputer()
numerical_train_y = train.select_dtypes(include=[np.number])
numerical_test_y = test.select_dtypes(include=[np.number])
imputed_train_y = pd.DataFrame(my_imputer.fit_transform(numerical_train_y))
imputed_test_y = pd.DataFrame(my_imputer.fit_transform(numerical_test_y))

msno.bar(imputed_train_y.sample(100))


train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)