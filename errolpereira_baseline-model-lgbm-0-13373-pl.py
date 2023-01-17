# Importing packages

import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import lightgbm as lgb

import xgboost as xgb



# Warnings

import warnings

warnings.filterwarnings('ignore')
# Reading data

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Getting the number of continuous and categorical variables

cat_cols = [x for x in train_df.columns if train_df[x].dtype == 'object']

cont_cols = [x for x in train_df.columns if train_df[x].dtype != 'object']



# Appending categorical columns and removing continous columns

cat_cols.extend(['MSSubClass', 'OverallQual', 'OverallCond'])

for x in ['MSSubClass', 'OverallQual', 'OverallCond']:

    cont_cols.remove(x)
# Categorical columns with missing values for training data

train_miss_cat = []

for x in cat_cols:

    if train_df[x].isnull().sum() > 0:

        train_miss_cat.append(x)



# Continuous cols with missing values for trainin data

train_miss_cont = []

for x in cont_cols:

    if train_df[x].isnull().sum() > 0:

        train_miss_cont.append(x)

        

# Categorical columns with missing values for test data

test_miss_cat = []

for x in cat_cols:

    if test_df[x].isnull().sum() > 0 and x not in train_miss_cat:

        test_miss_cat.append(x)



# Continuous cols with missing values for test data

test_miss_cont = []

for x in cont_cols:

    if x != 'SalePrice':

        if test_df[x].isnull().sum() > 0 and x not in train_miss_cont:

            test_miss_cont.append(x)



print(f'Missing columns in train and test {train_miss_cat+train_miss_cont}\n')

print(f'Additional Missing columns only in test {test_miss_cat+test_miss_cont}')
# If Garage is not present in a house then all the values related to Garage should be null or 0

grg_built = train_df.loc[train_df.GarageYrBlt.isnull()]

print(grg_built.shape)

grg_built[['GarageYrBlt', 'GarageType', 'GarageCars', 'GarageArea', 'GarageCond', 'GarageFinish']].isnull().sum()
# If MsnVnrType is None then all the values in MsnVnrArea should be null or 0

msn_area = train_df.loc[train_df.MasVnrArea.isnull()]

print(msn_area.shape)

msn_area[['MasVnrArea', 'MasVnrType']].isnull().sum()
# If Basement is NA then all the values for Basement related colummns be null or 0

bsnt_df = test_df.loc[test_df.BsmtFullBath.isnull()]

print(bsnt_df.shape)

bsnt_df[['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']].head()
# Function to impute the missing values in train and test data.

def impute_missing_values(train_miss_cat, train_miss_cont, test_miss_cat, test_miss_cont):

    # Handling missing values in cat columns for train and test data

    for column in train_miss_cat:

        if column == 'MasVnrType':

            train_df[column].fillna('None', inplace=True)

            test_df[column].fillna('None', inplace=True)

        elif column == 'Electrical':

            train_df[column].fillna(train_df[column].mode()[0], inplace=True)

            test_df[column].fillna(test_df[column].mode()[0], inplace=True)

        else:

            train_df[column].fillna("NA", inplace=True)

            test_df[column].fillna("NA", inplace=True)

            

    # Handling NaNs in continuous cols for train and test

    for column in train_miss_cont:

        if column == 'MasVnrArea' or column == 'GarageYrBlt':

            train_df[column].fillna(0, inplace=True)

            test_df[column].fillna(0, inplace=True)

        else:

            train_df[column].fillna(train_df[column].median(), inplace=True)

            test_df[column].fillna(test_df[column].median(), inplace=True)

            

    # Handling NaNs in cat and cont cols for test

    for column in test_miss_cat:

        test_df[column].fillna(test_df[column].mode()[0], inplace=True)

    for column in test_miss_cont:

        test_df[column].fillna(0, inplace=True)        
# Imputing Nulls

impute_missing_values(train_miss_cat, train_miss_cont, test_miss_cat, test_miss_cont)
# Combining train and test data

data = pd.concat([train_df, test_df])



# Initializing the labelencoder

le = LabelEncoder()



for column in cat_cols:

    if cat_cols != 'Id':

        data[column] = le.fit_transform(data[column])



# Getting the train and test data back

train_df = data.iloc[:train_df.shape[0], :]

test_df = data.iloc[train_df.shape[0]:, :-1]
# Creating X and y variables

X = train_df.drop(['Id', 'SalePrice'], axis=1).values

y = train_df.SalePrice.values
# Splitting data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Creating a Baseline Model

linear_reg = LinearRegression()



# Fitting the model with data

linear_reg.fit(X_train, y_train)



# Making predictions on val dataser

y_pred = linear_reg.predict(X_test)



# Evaluating performance

print(f'Root Mean Squared Error : {np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))}')

print(f'Mean Absolute Error : {np.sqrt(mean_absolute_error(y_test, y_pred))}')

print(f'R2 Score : {np.sqrt(r2_score(y_test, y_pred))}')

print(f'Adjusted R2 Score : {1-(1-r2_score(y_test, y_pred))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))}')
### Using KFold validation technique



# Initializing the fold

kf = KFold(n_splits=10)



# Score

score = []



for train_index, test_index in kf.split(X):

    

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    # Initializing the model

    linear_reg = LinearRegression(normalize=True)



    # Fitting the model with data

    linear_reg = linear_reg.fit(X_train, y_train)



    # Predicting on test data

    y_pred = linear_reg.predict(X_test)



    score.append(np.sqrt(mean_squared_error(np.log(y_test), np.log(abs(y_pred)))))



print(f'Mean RMSE: {np.mean(score)}')
### Using KFold validation technique



# Initializing the fold

kf = KFold(n_splits=10)



# Score

score = []



for train_index, test_index in kf.split(X):

    

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    # Initializing the model

    rf_reg = RandomForestRegressor()



    # Fitting the model with data

    rf_reg = rf_reg.fit(X_train, y_train)



    # Predicting on test data

    y_pred = rf_reg.predict(X_test)



    score.append(np.sqrt(mean_squared_error(np.log(y_test), np.log(abs(y_pred)))))



print(f'Mean RMSE: {np.mean(score)}')
### Using KFold validation technique



# Initializing the fold

kf = KFold(n_splits=10)



# Score

score = []



for train_index, test_index in kf.split(X):

    

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    # Initializing the model

    lgbm_reg = lgb.LGBMRegressor()



    # Fitting the model with data

    lgbm_reg = lgbm_reg.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)



    # Predicting on test data

    y_pred = lgbm_reg.predict(X_test)



    score.append(np.sqrt(mean_squared_error(np.log(y_test), np.log(abs(y_pred)))))



print(f'Mean RMSE: {np.mean(score)}')
### Using KFold validation technique



# Initializing the fold

kf = KFold(n_splits=10)



# Score

score = []



for train_index, test_index in kf.split(X):

    

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    # Initializing the model

    xgb_reg = xgb.XGBRegressor()



    # Fitting the model with data

    xgb_reg = xgb_reg.fit(X_train, y_train)



    # Predicting on test data

    y_pred = xgb_reg.predict(X_test)



    score.append(np.sqrt(mean_squared_error(np.log(y_test), np.log(abs(y_pred)))))



print(f'Mean RMSE: {np.mean(score)}')
# Training on the entire data and making prediction file

linear_reg.fit(X, y)



# Predictions on test data

sub_pred = linear_reg.predict(test_df.drop('Id', axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = sub_pred
# Converting to csv file

sub_df.to_csv('LinearRegressionBaseModel.csv', index=None, header=True)
# Training on the entire data and making prediction file

rf_reg.fit(X, y)



# Predictions on test data

sub_pred = rf_reg.predict(test_df.drop('Id', axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = abs(sub_pred)



# Converting to csv file

sub_df.to_csv('RandomForestBaseModel.csv', index=None, header=True)
# Training on the entire data and making prediction file

lgbm_reg.fit(X, y)



# Predictions on test data

sub_pred = lgbm_reg.predict(test_df.drop('Id', axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = abs(sub_pred)



# Converting to csv file

sub_df.to_csv('LGBMBaseModel.csv', index=None, header=True)
# Training on the entire data and making prediction file

xgb_reg.fit(X, y)



# Predictions on test data

sub_pred = xgb_reg.predict(test_df.drop('Id', axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = abs(sub_pred)



# Converting to csv file

sub_df.to_csv('XGBBaseModel.csv', index=None, header=True)