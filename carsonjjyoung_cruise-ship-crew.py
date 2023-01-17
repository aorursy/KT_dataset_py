# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



X = pd.read_csv("../input/cruise_ship_info.csv", index_col = "Ship_name")



X.head()



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
y = X.crew

X.drop(["crew"], axis = 1, inplace = True)



print("%.2f" % y.mean())
X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                     train_size = 0.8, test_size = 0.2,

                                                     random_state = 0)


encoder = OneHotEncoder(handle_unknown = "ignore", sparse = False)



OH_cols_train = pd.DataFrame(encoder.fit_transform(X_train[["Cruise_line"]]))

OH_cols_valid = pd.DataFrame(encoder.transform(X_valid[["Cruise_line"]]))



OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



num_train = X_train.drop("Cruise_line", axis = 1)

num_valid = X_valid.drop("Cruise_line", axis = 1)



OH_X_train = pd.concat([num_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_valid, OH_cols_valid], axis=1)
OH_X_train.head()
# Define the model

crew_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05) # Your code here



# Fit the model

crew_model.fit(OH_X_train, y_train,

               early_stopping_rounds = 5,

               eval_set = [(OH_X_valid, y_valid)],

               verbose = False)



# Get predictions

crew_predictions = crew_model.predict(OH_X_valid) # Your code here



# Calculate MAE

mae = mean_absolute_error(crew_predictions, y_valid) # Your code here



print("Mean Absolute Error: %.2f" % mae)

print("Accuracy: %.2f%%" % (100 * (1 - (mae / y.mean()))))