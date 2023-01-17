# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

house_data = pd.read_csv("../input/housetrain.csv")
# Any results you write to the current directory are saved as output.
house_data.head()
house_data.describe()
columns_data = house_data.columns
#house_data.SalePrice
x = house_data[['LotArea', 'PoolArea']]
y = house_data.SalePrice

from sklearn.tree import DecisionTreeRegressor
house_model = DecisionTreeRegressor()
house_model.fit(x,y)

predicted_prices = house_model.predict(x)
y.head()
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, predicted_prices)
mae


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)

house_model.fit(train_X,train_y)

predictions = house_model.predict(val_X)

mae = mean_absolute_error(val_y,predictions)

mae
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
for max_leaf_nodes in [2,5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
house_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, house_preds))
print(house_data.isnull().sum())
data_without_missing_values = house_data.dropna(axis=1)
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(house_data)
house_data.dtypes.sample(5)
one_hot_encoded_training_predictors = pd.get_dummies(house_data)
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)


