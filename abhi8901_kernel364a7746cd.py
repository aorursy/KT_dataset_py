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
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *



it_file_path = "../input/house-prices-advanced-regression-techniques/train.csv"

home_data = pd.read_csv(it_file_path)

y=home_data.SalePrice

features=["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd", "KitchenAbvGr"]

X=home_data[features]



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



#for decisiontree

it_model = DecisionTreeRegressor(random_state=1)

it_model.fit(train_X, train_y)



val_predictions=it_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



#using best value for maximum leaf-nodes

it_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

it_model.fit(train_X, train_y)



val_predictions=it_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



#Random forest algorithm

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

val_predictions=rf_model.predict(val_X)

val_mae=mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X,y)
#Make predictions on the test file

test_data_path = "../input/house-prices-advanced-regression-techniques/test.csv"

test_data = pd.read_csv(test_data_path)



test_X = test_data[features]

test_prds = rf_model_on_full_data.predict(test_X)



output = pd.DataFrame({'Id':test_data.Id,

                      'SalePrice':test_prds})

output.to_csv('submission.csv', index=False)