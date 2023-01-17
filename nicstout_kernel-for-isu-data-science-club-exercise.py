# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
training_data.head()
X = training_data.select_dtypes(exclude='object')
X = X.dropna()

y = X.SalePrice

X = X.drop(labels="SalePrice", axis=1)
X.describe()
ames_model = DecisionTreeRegressor(random_state=1)
ames_model.fit(X.iloc[200:], y.iloc[200:])
print("Making predictions for the following 5 houses:")
print(X.iloc[200:205])
print("The predictions are")
print(ames_model.predict(X.iloc[200:205]))
print("Actual values:")
print(y.iloc[200:205])
from sklearn.metrics import mean_absolute_error #As someone who's most experienced with C and java, this line physically hurts me

predicted_home_prices = ames_model.predict(X.iloc[200:])
mean_absolute_error(y.iloc[200:], predicted_home_prices)
predicted_home_prices = ames_model.predict(X[:200])
mean_absolute_error(y[:200], predicted_home_prices)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
ames_model = DecisionTreeRegressor()
ames_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = ames_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.preprocessing.imputation import Imputer

# make copy to avoid changing original data (when Imputing)
new_data = training_data.copy().select_dtypes(exclude='object')

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))

new_y = new_data[40] # The imputer added two columns, I don't know why...
new_data = new_data.drop(labels=40,axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
y = training_data.SalePrice
X = training_data.drop(labels="SalePrice", axis=1)
one_hot_encoded_training_predictors = pd.get_dummies(X)
new_data = one_hot_encoded_training_predictors.copy()

cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))

train_X, val_X, train_y, val_y = train_test_split(new_data, y, random_state = 0)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=100, learning_rate=0.02)
my_model.fit(train_X, train_y, early_stopping_rounds=2, 
             eval_set=[(val_X, val_y)], verbose=False)

predictions = my_model.predict(val_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))