# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex4 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(font_scale=1)
# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test = pd.read_csv('../input/test.csv', index_col='Id')
#We will use them later to split the whle dataset

trainrow=X_full.shape[0]

testrow=X_test.shape[0]

print(trainrow, testrow)
testing=[i for i in X_full.columns if not i in X_test.columns]

print(testing)
correlation=X_full.corr().sort_values(by='SalePrice',ascending=False).round(3)

print(correlation['SalePrice'])   
sns.scatterplot(x='GrLivArea',y='SalePrice',data=X_full)
#dealing with 2 outliers

X_full=X_full.drop(X_full.loc[(X_full['GrLivArea']>4000) & (X_full['SalePrice']<200000)].index,0)

X_full.reset_index(drop=True, inplace=True)
missing1 = X_full.isnull().sum().sort_values(ascending=False)

missing1 = missing1.drop(missing1[missing1==0].index)

missing1 
categorical = [cname for cname in X_full.columns if

                    X_full[cname].dtype == "object"]

print(categorical)
missing_cat_features = ['PoolQC','MiscFeature','Alley', 'FireplaceQu', 'Fence', 

                    'GarageType', 'GarageFinish', 'GarageQual','GarageCond',

                        'BsmtQual', 'BsmtCond', 'Electrical', 'BsmtExposure', 

                        'BsmtFinType1', 'BsmtFinType2','MasVnrType']

X_full[missing_cat_features] = X_full[missing_cat_features].fillna("NA")

missing3 = X_full.isnull().sum().sort_values(ascending=False)

missing3 = missing3.drop(missing3[missing3==0].index)

missing3
X_full['LotFrontage']=X_full['LotFrontage'].fillna(X_full['LotFrontage'].dropna().median())

X_full['MasVnrArea']=X_full['MasVnrArea'].fillna(0)

X_full['GarageYrBlt']=X_full['GarageYrBlt'].fillna(0)
X_test[missing_cat_features] = X_test[missing_cat_features].fillna("NA")

missing_test = X_test.isnull().sum().sort_values(ascending=False)

missing_test = missing_test.drop(missing_test[missing_test==0].index)

print(missing_test) 



categorical_test = [cname for cname in X_test.columns if

                    X_test[cname].dtype == "object"]

categorical_test=[i for i in categorical_test if not i in missing_cat_features]

print(categorical_test)
extra_features=['MSZoning',  'Utilities', 'Exterior1st', 'KitchenQual', 'Functional',  'SaleType', 'Exterior2nd']

X_test['LotFrontage']=X_test['LotFrontage'].fillna(X_full['LotFrontage'].dropna().median())

X_test['MasVnrArea']=X_test['MasVnrArea'].fillna(0)

X_test['GarageYrBlt']=X_test['GarageYrBlt'].fillna(0)

num_feat = ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 

            'BsmtFinSF1', 'GarageArea', 'GarageCars', ]



X_test[extra_features] = X_test[extra_features].fillna("NA")

X_test[num_feat]=X_test[num_feat].fillna(0)



missing_test = X_test.isnull().sum().sort_values(ascending=False)

missing_test = missing_test.drop(missing_test[missing_test==0].index)

print(missing_test)
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)
print(X_full.columns)



X_full['Bath']=X_full['BsmtFullBath']+X_full['BsmtHalfBath']/2 + X_full['FullBath']+X_full['HalfBath']/2

X_full=X_full.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],1)



X_test['Bath']=X_test['BsmtFullBath']+X_test['BsmtHalfBath']/2 + X_test['FullBath']+X_test['HalfBath']/2

X_test=X_test.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],1)
X_full['GrLivArea_2']=X_full['GrLivArea']**2



X_full['TotalBsmtSF_2']=X_full['TotalBsmtSF']**2



X_full['GarageCars_2']=X_full['GarageCars']**2



X_full['1stFlrSF_2']=X_full['1stFlrSF']**2



X_full['GarageArea_2']=X_full['GarageArea']**2



X_full['Bath2']=X_full['Bath']**2
X_test['GrLivArea_2']=X_test['GrLivArea']**2



X_test['TotalBsmtSF_2']=X_test['TotalBsmtSF']**2



X_test['GarageCars_2']=X_test['GarageCars']**2



X_test['1stFlrSF_2']=X_test['1stFlrSF']**2



X_test['GarageArea_2']=X_test['GarageArea']**2



X_test['Bath2']=X_test['Bath']**2
X_test['Floors']=X_test['1stFlrSF']+X_test['2ndFlrSF']

X_test=X_test.drop(['1stFlrSF','2ndFlrSF'],1)



X_full['Floors']=X_full['1stFlrSF']+X_full['2ndFlrSF']

X_full=X_full.drop(['1stFlrSF','2ndFlrSF'],1)
# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 12 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

# Define model

model = RandomForestRegressor(n_estimators=780, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
from xgboost import XGBRegressor



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

# Define model

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)



# Bundle preprocessing and modeling code in a pipeline

mod = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

mod.fit(X_train, y_train)

# Preprocessing of validation data, get predictions

preds = mod.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
# Preprocessing of test data, fit model

preds_test = mod.predict(X_test) # Your code here



# Check your answer

step_2.check()
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
preds_test