from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor



# Read the datasets, set to variables

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')
#Removes rows missing the target

X.dropna(axis=0, subset=['SalePrice'], inplace=True)



# Set our target to a classic variable name like 'y'

y = X.SalePrice



# Remove our target from the 'X' variables

X.drop(['SalePrice'], axis=1, inplace=True)
# Train, Test, Split the dataset

# 80% for learning, 20% for validation (can change these params)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)



# Select numeric columns - save to a list with the variable name "numeric_cols"

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# Return a list of categorical columns/variables with "low cardinality"

# What is classed as low? It's arbitrary, you can play around with this.

# Below we set this to "< 10" *** YOU CAN TRY DIFFERENT VALUES FOR THIS PARAMETER WHEN "TUNING" YOUR MODEL ***

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
# Add the numerical and low-cardinality lists together

my_cols = low_cardinality_cols + numeric_cols



# Update the X_train, X_valid and X_test lists to only contain the selected columns

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)



X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
# Define our first model with just the default parameters

my_model_1 = XGBRegressor(random_state=0)
# Fit the model to the training set.

my_model_1.fit(X_train, y_train)
# Get predictions for "y", given the X_valid dataset

predictions_1 = my_model_1.predict(X_valid)
# Calculate MAE

mae_1 = mean_absolute_error(predictions_1, y_valid)

# Display the MAE

print("Mean Absolute Error:" , mae_1)
# Define the model

my_model_2 = XGBRegressor(random_state=0, n_estimators=1000, learning_rate=0.05)
# Fit the model

my_model_2.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)], verbose=False)
# Let's predict

predictions_2 = my_model_2.predict(X_valid)



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2, y_valid)

# Print MAE

print("Mean Absolute Error:" , mae_2)
# Predict the target from the test dataset.

test_predict = my_model_2.predict(X_test)



# Output our predictions to a CSV file

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': test_predict})

output.to_csv('submission.csv', index=False)