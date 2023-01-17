# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
    

# Any results you write to the current directory are saved as output.
#training data

home_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
home_data.head()
#test data

home_data_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
home_data.columns

y = home_data.SalePrice
home_data['SalePrice'].describe()
features = ['TotRmsAbvGrd', 'GrLivArea', 'KitchenAbvGr',  'BedroomAbvGr','OpenPorchSF', 'OverallQual']
x = home_data[features]

x.describe()
x.head()
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state =0)
model = DecisionTreeRegressor(random_state =0)
model.fit(train_x, train_y)
val_prediction = model.predict(val_x)
val_mae =mean_absolute_error(val_prediction, val_y)
print("Validation MAE: {}".format(val_mae))
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
print("\nSetup complete")
def get_max(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model =DecisionTreeRegressor(max_leaf_nodes =max_leaf_nodes, random_state =0)
    model.fit(train_x, train_y)
    preds_val =model.predict(val_x)
    mae =mean_absolute_error(val_y, preds_val)
    return (mae)
    


for max_leaf_nodes in [20, 50, 100, 200, 500, 5000,10000, 20000]:
    my_mae = get_max(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("max_leaf_nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state =0)
clf.fit(train_x, train_y)
home_preds = clf.predict(val_x)
print(mean_absolute_error(val_y, home_preds))
#test data preprocessing before predictions

test_x = home_data_test[features]
prediction_with_test = model.predict(test_x)
#prediction with our test data

print(prediction_with_test)
submission_data = pd.DataFrame({"Id": home_data_test.Id, "SalePrice": prediction_with_test})
submission_data.to_csv("submission.csv", index=False)

