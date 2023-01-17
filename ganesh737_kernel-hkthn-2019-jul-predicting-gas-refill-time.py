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



# Code for Regression algo and testing

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
# Read the data

train_data = pd.read_csv("../input/Training.csv")

test_data = pd.read_csv("../input/Testing.csv")



# Target result

y = train_data.EndTime

features = ['Cycle', 'Type', 'Time', 'Values']

X = train_data[features]



# Break off validation set from training data set

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# X_train, X_valid, y_train, y_valid = train_test_split(X, y)
# Decision Tree Regressor

dtr_model = DecisionTreeRegressor()



# Fit Model

dtr_model.fit(X_train, y_train)



# Make predictions

val_pred_dtr = dtr_model.predict(X_valid)



# MAE

val_mae = mean_absolute_error(val_pred_dtr, y_valid)



print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
# Linear Regressor

lr_model = LinearRegression()



# Fit Model

lr_model.fit(X_train, y_train)



# Make predictions

val_pred_lr = lr_model.predict(X_valid)



# MAE

val_mae = mean_absolute_error(val_pred_lr, y_valid)



print("Validation MAE: {:,.0f}".format(val_mae))
# Random Forest Regressor

rf_model = RandomForestRegressor(random_state=1)



# Fit Model

rf_model.fit(X_train, y_train)



# Make predictions

val_pred_rf = rf_model.predict(X_valid)



# MAE

val_mae = mean_absolute_error(val_pred_rf, y_valid)



print("Validation MAE: {:,.0f}".format(val_mae))
val_pred_dtr
val_pred_lr
val_pred_rf
y_valid