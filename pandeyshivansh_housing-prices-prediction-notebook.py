# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from sklearn.model_selection import train_test_split

# Reading the data
X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Removing rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Breaking off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Selecting numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keeping selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, using pandas to shorten the code)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
X.head()
from xgboost import XGBRegressor

# Define the model
my_model = XGBRegressor(random_state=0)

# Fit the model
my_model.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error

# Get predictions
predictions = my_model.predict(X_valid)
# Calculate MAE
mae = mean_absolute_error(y_valid,predictions)

print("Mean Absolute Error:" , mae)
my_model = XGBRegressor(random_state=0, n_estimators=500, learning_rate=0.05)

# Fitting the model
my_model.fit(X_train,y_train)

# Get predictions
predictions = my_model.predict(X_valid)

# Calculate MAE
mae = mean_absolute_error(y_valid,predictions)

print("Mean Absolute Error:" , mae)
X_final = X[my_cols].copy()
X_test = X_test_full[my_cols].copy()

X_final = pd.get_dummies(X_final)
X_test = pd.get_dummies(X_test)
X_final, X_test = X_final.align(X_test, join='left', axis=1)
my_model.fit(X_final,y)
preds_test = my_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)