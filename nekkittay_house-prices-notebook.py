import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import os
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train_data.head()
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

x = train_data[features]

y = train_data.SalePrice

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8,test_size=0.2,random_state=0)

model = RandomForestRegressor(n_jobs=-1 ,random_state=1)

model.fit(x_train, y_train)
score = model.score(x_valid, y_valid)

print(score)
test = test_data[features]

test_pred = model.predict(test)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_pred})

output.to_csv('submission.csv', index=False)

print(output)