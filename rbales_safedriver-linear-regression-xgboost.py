import pandas as pd

import numpy as np

import random



pd.options.mode.chained_assignment = None  # default='warn'
def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    

    return np.sqrt(np.mean(calc))



def gini(list_of_values):

    sorted_list = sorted(list(list_of_values))

    height, area = 0, 0

    for value in sorted_list:

        height += value

        area += height - value / 2.

    fair_area = height * len(list_of_values) / 2

    return (fair_area - area) / fair_area





def normalized_gini(y_pred, y):

    normalized_gini = gini(y_pred)/gini(y)

    return normalized_gini

    



predicted_y = np.random.randint(100, size = 1000)

desired_y = np.random.randint(100, size = 1000)



print (normalized_gini(predicted_y, desired_y))
# Load Training Data

df_train = pd.read_csv('data/train.csv', index_col = 'id')

print(df_train.shape)

df_train.head()
# Load Test Data

df_test = pd.read_csv('data/test.csv', index_col = 'id')

print(df_test.shape)

df_test.head()
# Split the Train DataSet into X and y

X = df_train.drop('target', axis=1)

y = df_train.target



# Shuffle and Split the data

# This is running a StratifiedShuffleSplit in sklearn

import sklearn.model_selection as skms

X_train, X_validation, y_train, y_validation = skms.train_test_split(X, y,

                                                                     test_size=0.2, train_size=0.8,

                                                                     random_state=42)
# Perform a LInear Regression

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
# Generate Metrics on Validation Set

from sklearn.metrics import mean_squared_error



y_prediction = model.predict(X_validation)

rmsle_val = rmsle(y_validation, y_prediction)

rmse_val = mean_squared_error(y_validation, y_prediction)**0.5

normalized_gini_val = normalized_gini(y_prediction, y_validation)



print('Validation Metrics')

print('Normalized gini:', normalized_gini_val)

print('Root Mean Squared Logarithmic Error:', rmsle_val)

print('Root Mean Squared Error:', rmse_val)
# Train Linear Regression model

from xgboost import XGBRegressor

model_xgb = XGBRegressor()

model_xgb.fit(X_train, y_train)
# Generate Metrics on Validation Set

from sklearn.metrics import mean_squared_error



y_prediction = model_xgb.predict(X_validation)

rmsle_val = rmsle(y_validation, y_prediction)

rmse_val = mean_squared_error(y_validation, y_prediction)**0.5

normalized_gini_val = normalized_gini(y_prediction, y_validation)



print('Validation Metrics')

print('Normalized gini:', normalized_gini_val)

print('Root Mean Squared Logarithmic Error:', rmsle_val)

print('Root Mean Squared Error:', rmse_val)
# Predict on the Test Dataset

X_test = df_test

y_test = model_xgb.predict(X_test)
# Build the Submission Dataset

predictions = pd.DataFrame()

predictions['id'] = X_test.index

predictions['target'] = y_test.tolist()



print(predictions.shape)

print(predictions.head())
# Save Output

import time

submission_path = 'data/submission_' + str(time.time()) + '.csv'

predictions.to_csv(submission_path, index=False)