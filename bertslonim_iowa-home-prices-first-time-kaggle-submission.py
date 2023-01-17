#setup notebook
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
# Read the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)

target = train.SalePrice
# concatenate the train and test data so they can be processed/transformed the same way and then separated back in train/test
train_len = len(train)
dataset = pd.concat([train.iloc[:,:-1], test], ignore_index=True)

# check to see that the concatenation looks correct
print(dataset.shape)
dataset.head(3)
# Identify features with high percentage of missing values
print(dataset.isnull().sum().sort_values(ascending=False)/len(dataset)*100)
# Drop features with >50% of values missing
dataset_reduced = dataset.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)

# confirm that the features have been dropped
dataset_reduced.shape
# replace missing numeric values with mean values for those columns
dataset_reduced.fillna(dataset_reduced.mean(), inplace=True)

# check
dataset_reduced.head()
# Identify columns with numeric values so they can be scaled
dataset_reduced_numeric_cols = [col for col in dataset_reduced.columns if dataset_reduced[col].dtype 
                                in ['int64', 'float64']]

# check
print(dataset_reduced_numeric_cols)
# exclude Id and YrSold from numeric features to be scaled
dataset_reduced_numeric_cols = [e for e in dataset_reduced_numeric_cols if e not in ('Id', 'YrSold')]

# check
print(dataset_reduced_numeric_cols)
# scale numeric features

# create a dataframe containing just numeric columns to be scaled
numeric_dataset = dataset_reduced[dataset_reduced_numeric_cols]

# apply StandardScaler to the new dataframe
myscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
numeric_dataset_scaled = myscaler.fit_transform(numeric_dataset)

# copy the scaled values into the appropriate columns
dataset_reduced[dataset_reduced_numeric_cols] = numeric_dataset_scaled

# copy and rename 
dataset_scaled = dataset_reduced

dataset_scaled.head(3)
# Now onehotencode categoricals using get_dummies

cols_categorical = [col for col in dataset_scaled.columns if dataset_scaled[col].dtype == 'object']
dataset_scaled_dummies = pd.get_dummies(dataset_scaled, prefix=cols_categorical, columns=cols_categorical, drop_first=True)
dataset_scaled_dummies.shape
# Now reconstruct the test and train data

# this will be used for model training and validation
train_processed = pd.concat([dataset_scaled_dummies[:train_len], train['SalePrice']], axis=1)

# this will be used to generate the final submission
test_processed = dataset_scaled_dummies[train_len:]

# confirm that the data looks correct
print(train_processed.shape)
print(test_processed.shape)
train_processed['SalePrice'].head()

# Create target object and call it y
y = train_processed['SalePrice']

# Create X
X = train_processed.iloc[:,:-1]

# Split into training and testing data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit the model
rf_model.fit(train_X,train_y)

# Make predictions and calculate the mean absolute error of the Random Forest model on the validation data
rf_preds = rf_model.predict(test_X)
rf_val_mae = mean_absolute_error(rf_preds, test_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

xgb_model = xgb.XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(train_X, train_y, verbose=False)

# Make predictions and calculate the mean absolute error of the xgb model on the validation data
xgb_preds = xgb_model.predict(test_X)
xgb_val_mae = mean_absolute_error(xgb_preds, test_y)

print("Validation MAE for XGBoost Model: {}".format(xgb_val_mae))

# tweak xgb model paramters

my_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)


# Make predictions and calculate the mean absolute error of the tweaked xgb model on the validation data
preds = my_model.predict(test_X)
xgb_tweaked_val_mae = mean_absolute_error(preds, test_y)

print("Validation MAE for tweaked XGBoost Model: {}".format(xgb_tweaked_val_mae))
# make predictions which we will submit. 
test_preds = my_model.predict(test_processed)

output = pd.DataFrame({'Id': test_processed.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
