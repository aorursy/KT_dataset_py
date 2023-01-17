#

# Read training data

#



import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Read csv file

train_data = pd.read_csv("../input/train.csv")



# Extract price column

y_train = train_data.SalePrice



# Extract all features

X_train = train_data.drop(['Id', 'SalePrice'], axis=1)



# Display traingin data

X_train.head(10)
#

# Replace missing values with the statistical values

#



# For numerical columns

medians = X_train.select_dtypes(include=['int64', 'float64']).mean()

for f in X_train.select_dtypes(include=['int64', 'float64']):

    # Replace with median

    X_train[f] = X_train[f].fillna(medians[f])



# For categorical columns

modes = X_train.select_dtypes(include=['object']).mode()

for f in X_train.select_dtypes(include=['object']):

    # Replace with mode (most frequent value)

    X_train[f] = X_train[f].fillna(modes[f][0])

    

# Transform categorical variables to dummy variables

X_train = pd.get_dummies(X_train)



# List of all features

features = X_train.columns



# Display the result of modification

X_train.head(10)
#

# Remove outlier samples

#



from sklearn.ensemble import IsolationForest

import collections



# Predict outliers

outlier_pred = IsolationForest(random_state=1).fit_predict(X_train)



# Remove outliers

X_train, y_train = X_train[outlier_pred == 1], y_train[outlier_pred == 1]

X_train.head(10)
#

# Find the model with best parameters

#



from sklearn.model_selection import GridSearchCV

import xgboost as xgb



# Parameter variation

parameters = {'max_depth':[2,4,8]}



# Grid search

gs = GridSearchCV(xgb.XGBRegressor(random_state=1, n_estimators=100, objective='reg:squarederror', eval_metric='mae'), 

                  parameters, cv=5, scoring='neg_mean_absolute_error')

gs.fit(X_train, y_train)



# Best estimator

reg = gs.best_estimator_

reg.get_params()
#

# Prediction

#



# Create regression model

reg.fit(X_train, y_train)



# read test data

test_data = pd.read_csv("../input/test.csv")



# Extract feature columns

X_test = test_data.drop('Id', axis=1)



# For numerical NaNs

for f in X_test.select_dtypes(include=['int64', 'float64']):

    # Replace with median

    X_test[f] = X_test[f].fillna(medians[f])



# For categorical NaNs

for f in X_test.select_dtypes(include=['object']):

    # Replace with mode (most frequent value)

    X_test[f] = X_test[f].fillna(modes[f][0])



# Transform categorical variables to dummy variables

X_test = pd.get_dummies(X_test)

    

# Remove additional columns

for f in X_test.columns:

    if f not in X_train.columns:

        X_test = X_test.drop(f, axis=1)



# Add nescessary columns

for f in X_train.columns:

    if f not in X_test.columns:

        X_test[f] = 0



# Predict

X_test = X_test[X_train.columns]

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': reg.predict(X_test)})

output.to_csv('submission.csv', index=False)



# output.describe()

plt_data = pd.DataFrame({'predict': reg.predict(X_train), 'actual': y_train})

sns.jointplot('predict', 'actual', data=plt_data)