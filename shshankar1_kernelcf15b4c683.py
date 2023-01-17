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
train_data = pd.read_csv("../input/train.csv")
train_data.describe()
train_y = train_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X = train_data[feature_columns]
from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
test_data = pd.read_csv("../input/test.csv")
true_price = pd.read_csv("../input/sample_submission.csv")
#print(true_price)
test_data_price = pd.merge(left=test_data,left_index=True,right=true_price,right_index=True,how="inner")
test_X = test_data_price[feature_columns]
test_y = test_data_price.SalePrice
print(test_X.head())
print(test_y.head())
predicted_price = iowa_model.predict(test_X)
print(predicted_price[:5])
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(test_y, predicted_price)
print(val_mae)