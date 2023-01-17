# Import necessary libraries

import datetime
import numpy as np
import pandas as pd
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries

from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
# Load and read files
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
train_df = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
test_df = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

# Rename columns
train_df.rename(columns={'Country_Region': 'Country'}, inplace=True)
train_df.rename(columns={'Province_State': 'State'}, inplace=True)
test_df.rename(columns={'Country_Region': 'Country'}, inplace=True)
test_df.rename(columns={'Province_State': 'State'}, inplace=True)

display(train_df.head(5))
display(test_df.head(5))
train_df.info()
print('\n')
test_df.info()
# Transform the normal date to pandas datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

display(train_df.head(5))
display(test_df.head(5))
# Shape of training data
print(train_df.shape)

# Number of missing values in each column of training data
missing_train_count_col = (train_df.isnull().sum())
print(missing_train_count_col[missing_train_count_col>0])

# Shape of testing data
print(test_df.shape)
# Number of missing values in each column of training data
missing_test_count_col = (test_df.isnull().sum())
print(missing_test_count_col[missing_test_count_col>0])
# Fill null values
train_df['State'].fillna('No State', inplace=True)
test_df['State'].fillna('No State', inplace=True)

# Number of missing values in each column of training data
missing_train_count_col = (train_df.isnull().sum())
print(missing_train_count_col[missing_train_count_col>0])

# Number of missing values in each column of training data
missing_test_count_col = (test_df.isnull().sum())
print(missing_test_count_col[missing_test_count_col>0])
print('\n')

# Double check no remaining missing values
train_df.info()
print('\n')
test_df.info()
# Apply Label Encoding to train and test data
train_df_encoded = train_df.copy()
test_df_encoded = test_df.copy()

# Initialize Label encoder
le = LabelEncoder()

# Create date time features
def create_time_features(df):
    df['date'] = df['Date']
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    return df

train_df_encoded = create_time_features(train_df_encoded)
test_df_encoded = create_time_features(test_df_encoded)
train_df_encoded.State = le.fit_transform(train_df_encoded.State)
train_df_encoded.Country = le.fit_transform(train_df_encoded.Country)
test_df_encoded.State = le.fit_transform(test_df_encoded.State)
test_df_encoded.Country = le.fit_transform(test_df_encoded.Country)

display(train_df_encoded.tail())
print('\n')
display(test_df_encoded.tail())
# Specify all features for prediction
x_features_drop = ['ConfirmedCases', 'Fatalities', 'Date', 'date']
y_target1 = ['ConfirmedCases']
y_target2 = ['Fatalities']

# Assign features into X, y1, y2 for training and testing
X = train_df_encoded.drop(x_features_drop, axis=1)
y1 = train_df_encoded[y_target1]
y2 = train_df_encoded[y_target2]

display(X.head())
display(y1.tail())
display(y2.tail())
# # Split into validaion and training data on 2 features
rft1_train_X, rft1_val_X, rft1_train_y, rft1_val_y = train_test_split(X, y1, train_size=0.8, test_size=0.2, random_state=1)
rft2_train_X, rft2_val_X, rft2_train_y, rft2_val_y = train_test_split(X, y2, train_size=0.8, test_size=0.2, random_state=2)

# Define the models
model_1 = DecisionTreeClassifier(splitter='best', max_features='log2', random_state=42)
model_2 = DecisionTreeClassifier(splitter='random', max_features='log2', random_state=42)
model_3 = DecisionTreeClassifier(splitter='best', max_features='sqrt', random_state=42)
model_4 = DecisionTreeClassifier(splitter='random', max_features='sqrt', random_state=42)
model_5 = DecisionTreeClassifier(splitter='random', max_features='log2', random_state=42)
model_6 = DecisionTreeClassifier(splitter='random', max_features='sqrt', random_state=42)
model_7 = DecisionTreeClassifier(splitter='best', max_features='log2', random_state=42)
model_8 = DecisionTreeClassifier(splitter='best', max_features='sqrt', random_state=42)

rf_models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8]

# Function for comparing different models
def score_model(model, train_X, val_X, train_y, val_y):
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    #accuracy = accuracy_score(y_v, preds)
    return mean_absolute_error(val_y, preds)

# Evaluate the models for y1:
for i in range(0, len(rf_models)):
    mae = score_model(rf_models[i], rft1_train_X, rft1_val_X, rft1_train_y, rft1_val_y)
    print('Model %d MAE y1: %d' % (i+1, mae))

print('\n')
    
# Evaluate the models for y2:
for i in range(0, len(rf_models)):
    mae = score_model(rf_models[i], rft2_train_X, rft2_val_X, rft2_train_y, rft2_val_y)
    print('Model %d MAE y2: %d' % (i+1, mae))
# # Split into validaion and training data on 2 features
# xgbt1_train_X, xgbt1_val_X, xgbt1_train_y, xgbt1_val_y = train_test_split(X, y1, train_size=0.8, test_size=0.2, random_state=1)

# # Fit and predict
# xgb = XGBRegressor(random_state=42)
# xgb.fit(xgbt1_train_X, xgbt1_train_y)
# xgb_predictions = xgb.predict(xgbt1_val_X)
# print(xgb_predictions)
# # Evaluate predictions
# xgb_accuracy = accuracy_score(xgbt1_val_y, xgb_round_predictions)
# print("Accuracy: %.2f%%" % (xgb_accuracy * 100.0))
# Choose best Random Forest Model for y1 and y2
best_rf_model_y1 = model_2
best_rf_model_y2 = model_2

# Assign features to test data
x_test_features_drop = ['Date', 'date']
X_test = test_df_encoded.drop(x_test_features_drop, axis=1)

# Predict the best model for y1 and y2
y1_pred = best_rf_model_y1.predict(X_test)
y2_pred = best_rf_model_y2.predict(X_test)

print(y1_pred[100:150])
print(y2_pred[100:150])
# Save predictions in format used for competition scoring
output = pd.DataFrame({'ForecastId': test_df.ForecastId, 'ConfirmedCases': rnd_y1_pred, 'Fatalities': rnd_y2_pred})
output.to_csv('submission.csv', index=False)
print(output.tail(10))
print('Submission file successfully saved..')