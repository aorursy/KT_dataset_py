import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

%matplotlib inline

import matplotlib.pyplot as plt
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
# Remove rows with missing target, separate target from predictors

train_df.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_df.SalePrice              

train_df.drop(['SalePrice'], axis=1, inplace=True)
plt.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":y, "log(price + 1)":np.log1p(y)});

prices.hist();
y = np.log1p(y)
from sklearn.model_selection import train_test_split

# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_df, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test_df[my_cols].copy()
# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error



# Define the model

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05) # Your code here



# Fit the model

my_model.fit(X_train, y_train,

               early_stopping_rounds=5,

              eval_set=[(X_valid, y_valid)],

              verbose=0)



# Get predictions

predictions = my_model.predict(X_valid) # Your code here



# undo the log1p transform using expm1, and get the Mean Absolute Error

print("Mean Absolute Error:" , np.sqrt(mean_squared_error(np.expm1(predictions), np.expm1(y_valid))))
# Preprocessing of test data, fit model

preds_test = np.expm1(my_model.predict(X_test))
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)