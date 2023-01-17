import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

import numpy as np

from itertools import product

from sklearn.preprocessing import LabelEncoder



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

import time

import sys

import gc

import pickle

sys.version_info
X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')
X_full
X_full[['SalePrice']].idxmax() 
#outliers = [30, 88, 462, 631, 1322]

#X_full.drop(X_full.index[outliers])
columns=['Alley','LandContour','Fence','GarageYrBlt','HalfBath','Condition1','Condition2','PavedDrive','MiscFeature']
X_full=X_full.drop(columns,axis=1)

X_test_full=X_test_full.drop(columns,axis=1)
# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)
# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() <7  and 

                    X_train_full[cname].dtype == "object"]
# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]
# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head()
X_train.MSZoning.unique()
plt.figure(figsize=(16,9))

sns.barplot(x=X_train.MSZoning,y=y_train)
plt.figure(figsize=(16,9))

sns.barplot(x=X_train.index,y=y_train)
#common regression line

plt.figure(figsize=(16,9))

sns.regplot(x=X_train['YearRemodAdd'], y=y_train)
plt.figure(figsize=(16,9))

sns.barplot(x=X_train.RoofStyle,y=y_train)
#common regression line

plt.figure(figsize=(16,9))

sns.regplot(x=X_train['YearBuilt'], y=y_train)
y_train.max()
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
# Define model

import xgboost 

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=500,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 

lightgbm = LGBMRegressor(objective='regression',

                         num_leaves=4,

                         learning_rate=0.01,

                         n_estimators=5000,

                         max_bin=200,

                         bagging_fraction=0.75,

                         bagging_freq=5,

                         bagging_seed=7,

                         feature_fraction=0.2,

                         feature_fraction_seed=7,

                         verbose=-1,

                         # min_data_in_leaf=2,

                         # min_sum_hessian_in_leaf=11

                         )

# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])
ts=time.time()

# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))

time.time()-ts
# Preprocessing of test data, fit model

preds_test = clf.predict(X_test)

preds_test=preds_test.astype('int64')
plot_features(model,(16,20))
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)