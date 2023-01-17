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

import matplotlib.pyplot as plt



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



# Initializing the encoder

le = LabelEncoder()



for column in cat_cols:

    if column != 'Id':

        data[column] = le.fit_transform(data[column])





# Getting the train and test data back

train_df = data.iloc[:train_df.shape[0], :]

test_df = data.iloc[train_df.shape[0]:, :-1]
# Making a copy

train_og = train_df.copy()

test_og = test_df.copy()
# Utilities in train and test data

print(train_df.Utilities.value_counts())

print(test_df.Utilities.value_counts())
# Mainiting a list of all the features to drop

drop_list = ['Utilities']

columns = train_df.drop(drop_list, axis=1).columns.tolist()
# Function to create frequency group

def frequency_group(min_threshold=1, col='MSSubClass'):

    # Making groups of nominal category

    data_df[f"{col}_group"] = np.where(data_df[f'{col}_Frequency'] <= min_threshold, '0', data_df[col])
# Combining train and test data

data_df = pd.concat([train_df, test_df])



# Frequency Groups for MSSubClass

data_df['MSSubClass_Frequency'] =  data_df.groupby(['MSSubClass'])['MSSubClass'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=2)



# Frequency Groups for Neighborhood

data_df['Neighborhood_Frequency'] =  data_df.groupby(['Neighborhood'])['Neighborhood'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=2, col='Neighborhood')



# Frequency Groups for Condition1

data_df['Condition1_Frequency'] =  data_df.groupby(['Condition1'])['Condition1'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=1.5, col='Condition1')



# Frequency Groups for Condition2

data_df['Condition2_Frequency'] =  data_df.groupby(['Condition2'])['Condition2'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=1.5, col='Condition2')



# Frequency Groups for RoofMatl

data_df['RoofMatl_Frequency'] =  data_df.groupby(['RoofMatl'])['RoofMatl'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=1.5, col='RoofMatl')



# Frequency Groups for Exterior1st

data_df['Exterior1st_Frequency'] =  data_df.groupby(['Exterior1st'])['Exterior1st'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=2, col='Exterior1st')



# Frequency Groups for Exterior2nd

data_df['Exterior2nd_Frequency'] =  data_df.groupby(['Exterior2nd'])['Exterior2nd'].transform('count') / len(data_df) * 100

frequency_group(min_threshold=2, col='Exterior2nd')



# Dropping the attr3 and attr4 cols

drop_list.extend(['MSSubClass', 'Neighborhood', 'Condition1', 'Condition2',

                 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MSSubClass_Frequency',

                 'Neighborhood_Frequency', 'Condition1_Frequency', 'Condition2_Frequency',

                 'RoofMatl_Frequency','Exterior1st_Frequency', 'Exterior2nd_Frequency'])

data_df.drop(drop_list, axis=1, inplace=True)



# Renaming the group col created

col_rename = {

    'MSSubClass_group' : 'MSSubClass',

    'Neighborhood_group' : 'Neighborhood',

    'Condition1_group' : 'Condition1',

    'Condition2_group' : 'Condition2',

    'RoofMatl_group' : 'RoofMatl',

    'Exterior1st_group' : 'Exterior1st',

    'Exterior2nd_group' : 'Exterior2nd'

}

data_df.rename(columns=col_rename, inplace=True)

data_df = data_df[columns]
# Spliting the data back into train and test

train_df = data_df.iloc[:train_df.shape[0], :]

test_df = data_df.iloc[train_df.shape[0]:, :-1]
# Checking the impact of the above procedure on model performance



# Creating X and y variables

X = train_df.drop(['Id', 'SalePrice'], axis=1).values

y = train_df.SalePrice.values





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
# LightGBM model importance

plt.figure(figsize=(8.5, 10))

plt.title('Feature Importance LGBM model')

feat_importances = pd.Series(lgbm_reg.feature_importances_, index=train_df.drop(['Id', 'SalePrice'], axis=1).columns)

feat_importances.nsmallest(20).plot(kind='barh')

plt.show()
# Getting back the OG dataset

train_df = train_og.copy()

test_df = test_og.copy()
# Making new droplist

drop_list = ['Street', 'Utilities', 'Alley', 'Condition2', 'RoofMatl', 'Heating', 'Electrical',

            'GarageCond', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'BldgType', 'Id', 'SalePrice']



# Creating X and y variables

X = train_df.drop(drop_list, axis=1).values

y = train_df.SalePrice.values





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
# Training on the entire data and making prediction file

lgbm_reg.fit(X, y)



# Predictions on test data

sub_pred = lgbm_reg.predict(test_df.drop(drop_list[:-1], axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = abs(sub_pred)



# Converting to csv file

sub_df.to_csv('LGBMFeat1l.csv', index=None, header=True)
# Combining train and test data

data_df = pd.concat([train_df, test_df])



# Total Floor Area of the house

data_df['Total_Floor_Area'] = data_df['1stFlrSF'] + data_df['2ndFlrSF']
# Average OverallQuality of the houses in each neighborhood

average_neighborhood_qual = data_df.groupby(by=['Neighborhood'])['OverallQual'].aggregate('mean').to_frame('AvgQualByNeighborhood').reset_index()

average_neighborhood_qual.head()
# Average OverallCondition of the houses in each neighborhood

average_neighborhood_cond = data_df.groupby(by=['Neighborhood'])['OverallCond'].aggregate('mean').to_frame('AvgCondByNeighborhood').reset_index()

average_neighborhood_cond.head()
# Number of housed in the neigborhood

number_houses_neighborhood = data_df.groupby(by=['Neighborhood'])['Id'].aggregate('count').to_frame('NumHousesNeighborhood').reset_index()

number_houses_neighborhood.head()
# Combining the newly created features

neighborhood_feat = average_neighborhood_qual.merge(average_neighborhood_cond, on='Neighborhood', how='left')

neighborhood_feat = neighborhood_feat.merge(number_houses_neighborhood, on='Neighborhood', how='left')

data_df = data_df.merge(neighborhood_feat, on='Neighborhood', how='left')

data_df.head()
# Spliting the data back into train and test

columns = data_df.columns.tolist()

columns.remove('SalePrice')

columns.append('SalePrice')

data_df = data_df[columns]



train_df = data_df.iloc[:train_df.shape[0], :]

test_df = data_df.iloc[train_df.shape[0]:, :-1]
# Creating X and y variables

X = train_df.drop(['Id', 'SalePrice'], axis=1).values

y = train_df.SalePrice.values





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
# LightGBM model importance

plt.figure(figsize=(8.5, 10))

plt.title('Feature Importance LGBM model')

feat_importances = pd.Series(lgbm_reg.feature_importances_, index=train_df.drop(['Id', 'SalePrice'], axis=1).columns)

feat_importances.nsmallest(20).plot(kind='barh')

plt.show()
# Training on the entire data and making prediction file

lgbm_reg.fit(X, y)



# Predictions on test data

sub_pred = lgbm_reg.predict(test_df.drop('Id', axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = abs(sub_pred)



# Converting to csv file

sub_df.to_csv('LGBMFeat2.csv', index=None, header=True)
# Creating X and y variables

X = train_df.drop(['Id', 'SalePrice', 'AvgQualByNeighborhood', 'AvgCondByNeighborhood', 'NumHousesNeighborhood'], axis=1).values

y = train_df.SalePrice.values





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
# Training on the entire data and making prediction file

lgbm_reg.fit(X, y)



# Predictions on test data

sub_pred = lgbm_reg.predict(test_df.drop(['Id', 'AvgQualByNeighborhood', 'AvgCondByNeighborhood', 'NumHousesNeighborhood'], axis=1).values)



# Adding to the sub df

sub_df['SalePrice'] = abs(sub_pred)



# Converting to csv file

sub_df.to_csv('LGBMFeat3.csv', index=None, header=True)