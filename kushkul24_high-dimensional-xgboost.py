#importing packages

import pandas as pd

import numpy as np
#reading the data

train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')
#Missing value treatment

missing_train = train_data.isnull().sum()[train_data.isnull().sum()>0]

print('Original shape is: ', train_data.shape)

print(missing_train)



#To drop: Alley, PoolQC, Fence, MiscFeature, FireplaceQu
#Analyzing the missing value variables

train_data['MasVnrType'].unique() #replace with None

train_data['MasVnrArea'].unique() #replace with 0

train_data['BsmtCond'].unique() # drop nas for BsmtQual and BsmtCond



#checking of bsmtCond and BsmtExposure is related

train_data[train_data['BsmtCond'].isnull()]['BsmtExposure'].unique()

#All BestCond null values have null BsmtExposure, but one more Bsmt Exposure left

train_data[train_data['BsmtExposure'].isnull()]['BsmtCond'].unique() #- Remove this row



#BsmtFinType1 and BsmtFinType2

train_data[train_data['BsmtFinType2'].isnull()]['BsmtExposure'].unique() #- Drop this row, BsmtFinType1 will take care

#drop Electrical null row



#FireplaceQu

(train_data['FireplaceQu'].unique()) #to be removed



#LotFrontage

train_data.LotFrontage.value_counts() #replace by max



#GarageType

train_data['GarageType'].unique() #- Replace by 'None'

train_data['GarageYrBlt'].unique() #- Replace by Age with missing values as 1000 *See what to do here

train_data['GarageFinish'].unique() # Replace by 'None'

train_data['GarageQual'].unique() # Remove thid variable

train_data['GarageCond'].unique() # Remove this as well

#Removing nas of Garage columns instead of filling them

#Removing the variables we dont want

remove_cols = ['GarageQual', 'GarageCond', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']

for i in remove_cols:

    del train_data[i]

    del test_data[i]
#Taking care of missing values

train_data['MasVnrType'] = train_data['MasVnrType'].fillna(value = 'None')

train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(value = 0)

train_data.dropna(inplace = True, subset = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'Electrical'])



max_df = pd.DataFrame(train_data.LotFrontage.value_counts().sort_values(ascending = False))

max_df['value'] = max_df.index

max_LotFrintage = max_df.index[0]

train_data['LotFrontage'] = train_data['LotFrontage'].fillna(value = max_LotFrintage)



#Removing nas of Galarage columns

train_data.dropna(inplace = True, subset = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'BsmtFinType2'])
#Taking care of missing values in test data

test_data['MasVnrType'] = test_data['MasVnrType'].fillna(value = 'None')

test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(value = 0)

test_data.dropna(inplace = True, subset = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'Electrical'])



#Using traindata max value to fillna

test_data['LotFrontage'] = test_data['LotFrontage'].fillna(value = max_LotFrintage)



#Removing nas of Galarage columns

test_data.dropna(inplace = True, subset = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'BsmtFinType2'])

test_data.dropna(inplace = True)
#Checking if all null values are treated or not

missing_train = train_data.isnull().sum()[train_data.isnull().sum()>0]

print('Original shape is: ', train_data.shape)

print('Null values in train data: ', missing_train)

#We are good with null values



missing_test = test_data.isnull().sum()[test_data.isnull().sum()>0]

print('Original shape is: ', test_data.shape)

print('Null values in test data: ', missing_test)

#Strategy

#drop the columns with more than 50% missing values                       - Done

#drop the rows where y is missing in training                             - Done

#seperate the categorical and numerical columns                           - Done

#treat the categorical columns for missing values                         - Done

#treat the categorical columns with label encoder or oh encoder           - Done

#treat the numerical columns for missing values                           - Done

#Dropping the rows where y value is absent

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data.SalePrice

train_data.drop(['SalePrice'], axis = 1, inplace = True)

print(train_data.shape)
#Separating numeric and categorical columns

all_cols = train_data.columns

datatype = []

num_cols = []

cat_cols = []

for col in all_cols:

    if train_data[col].dtype in ['int64', 'float64']:

        num_cols.append(col)

    if train_data[col].dtype == 'O':

        cat_cols.append(col)

    #datatype.append(train_data[col].dtype)

#print(num_cols)

#print(cat_cols)



#Finding out cardinality of columns

col_cardinality = {}

for col in cat_cols:

    distinct_values = train_data[col].nunique()

    col_cardinality[col] = distinct_values



print(col_cardinality)

#cardinality seems low for all cols, using all as of now
#One hot encoding for categorical variables

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer



categorical_transformer = Pipeline(steps = [

        ('OHEncoding', OneHotEncoder(handle_unknown='ignore'))]) 



numerical_transformer = SimpleImputer(strategy='constant') #Does nothing as we have filled null values



preprocessor = ColumnTransformer(transformers = [

        ('num', numerical_transformer, num_cols),

        ('cat', categorical_transformer, cat_cols)

                                                ])



#defining the model

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators = 500, learning_rate = 500, random_state= 50)



#Working on the main step now

clf = Pipeline(steps = [

    ('preprocessor', preprocessor),

    ('model', my_model)   ])



clf.fit(train_data, y)

my_prediction = clf.predict(test_data)

output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': my_prediction})

output.to_csv('submission.csv', index=False)
