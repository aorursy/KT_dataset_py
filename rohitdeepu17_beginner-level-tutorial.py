import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from sklearn import preprocessing

import csv

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingRegressor

import lightgbm

#import parameters_housing

from sklearn.preprocessing import LabelBinarizer

from catboost import CatBoostClassifier

from catboost import CatBoostRegressor



import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

#%matplotlib inline

import seaborn as sns

print("Setup Complete")
train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

train_data.shape
test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

test_data.shape
train_data.columns
y = train_data['SalePrice']

y.shape
train_data.drop(['SalePrice'], axis=1,inplace=True)

train_data.shape
def get_list_of_columns_of_type(X,col_type):

    s = (X.dtypes == 'object')     #default categorical

    if col_type == 'categorical':

        s = (X.dtypes == 'object')

    elif col_type == 'numerical':

        s = (X.dtypes != 'object')

    cols = list(s[s].index)

    return cols
categorical_cols = get_list_of_columns_of_type(train_data, 'categorical')

print(len(categorical_cols))

categorical_cols
numerical_cols = get_list_of_columns_of_type(train_data, 'numerical')

print(len(numerical_cols))

numerical_cols
missing_val_count_num_train = (train_data[numerical_cols].isnull().sum())

missing_val_count_num_train.sort_values(ascending=False,inplace=True)

missing_val_count_num_train[missing_val_count_num_train > 0]
missing_val_count_cat_train = (train_data[categorical_cols].isnull().sum())

missing_val_count_cat_train.sort_values(ascending=False,inplace=True)

missing_val_count_cat_train[missing_val_count_cat_train > 0]
missing_val_count_num_test = (test_data[numerical_cols].isnull().sum())

missing_val_count_num_test.sort_values(ascending=False,inplace=True)

missing_val_count_num_test[missing_val_count_num_test > 0]
missing_val_count_cat_test = (test_data[categorical_cols].isnull().sum())

missing_val_count_cat_test.sort_values(ascending=False,inplace=True)

missing_val_count_cat_test[missing_val_count_cat_test > 0]
train_data_count = train_data.shape[0]

merged_data = train_data.append(test_data)

merged_data.shape
missing_val_count_num_merged = (merged_data[numerical_cols].isnull().sum())

missing_val_count_num_merged.sort_values(ascending=False,inplace=True)

missing_val_count_num_merged[missing_val_count_num_merged > 0]
missing_val_count_cat_merged = (merged_data[categorical_cols].isnull().sum())

missing_val_count_cat_merged.sort_values(ascending=False,inplace=True)

missing_val_count_cat_merged[missing_val_count_cat_merged > 0]
missing_val_count_cat_merged[missing_val_count_cat_merged > (merged_data.shape[0]/2)]
merged_data.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1,inplace=True)

merged_data.shape
numerical_cols = get_list_of_columns_of_type(merged_data,'numerical')

categorical_cols = get_list_of_columns_of_type(merged_data,'categorical')

numerical_cols
for col in numerical_cols:

        merged_data[col] = merged_data[col].fillna(merged_data[col].mean())
missing_val_count_num_merged = (merged_data[numerical_cols].isnull().sum())

missing_val_count_num_merged.sort_values(ascending=False,inplace=True)

missing_val_count_num_merged[missing_val_count_num_merged > 0]
for col in categorical_cols:

        merged_data[col] = merged_data[col].fillna("Unknown")
missing_val_count_cat_merged = (merged_data[categorical_cols].isnull().sum())

missing_val_count_cat_merged.sort_values(ascending=False,inplace=True)

missing_val_count_cat_merged[missing_val_count_cat_merged > 0]
label_encoder = LabelEncoder()

for col in categorical_cols:

    merged_data[col] = label_encoder.fit_transform(merged_data[col])
categorical_cols = get_list_of_columns_of_type(merged_data,'categorical')

categorical_cols
train_data = merged_data[0:train_data_count]

train_data.shape
test_data = merged_data[train_data_count:]

test_data.shape
X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2,random_state=0)

print(" Training Data Shape : {}\n Training Target Shape : {}\n Validation Data Shape : {}\n Validation Target Shape : {}".format(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape))
my_model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42) 
my_model.fit(X_train, y_train)
preds = my_model.predict(X_train)

mean_absolute_error(y_train, preds)
preds = my_model.predict(X_valid)

mean_absolute_error(y_valid, preds)
my_model.fit(train_data,y)
preds = my_model.predict(test_data)

preds.shape
submission = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")

submission['SalePrice'] = preds
submission.to_csv("submission.csv", index=False)