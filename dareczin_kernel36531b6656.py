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



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = [ 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'LotArea', 'TotRmsAbvGrd', 'BedroomAbvGr', 'MSSubClass', 'YrSold', 'PoolArea','MoSold', 'Fireplaces',  'GrLivArea',  'OverallQual', 'OverallCond']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(n_estimators=100,random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

model_6 = RandomForestRegressor(n_estimators=200, criterion='mae', random_state=0)



models = [model_1, model_2, model_3, model_4, model_5, model_6]









def score_model(model, X_t=train_X, X_v=val_X, y_t= train_y, y_v= val_y):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))
home_data.info()

X.isna().sum()


rf_model_on_full_data = RandomForestRegressor(n_estimators=100,random_state=1)

rf_model_on_full_data.fit(X, y)


test_data_path = '../input/test.csv'





test_data = pd.read_csv(test_data_path)







test_X = test_data[features]

test_preds = rf_model_on_full_data.predict(test_X)



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

print(output)

output.to_csv('submission.csv', index=False)




