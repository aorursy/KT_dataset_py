# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_dataset_path = '/kaggle/input/train.csv'
test_dataset_path = '/kaggle/input/test.csv'

import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
#Training Data-Set
X_full = pd.read_csv(train_dataset_path, index_col='Id')

#Tesing Data-Set
X_test_full = pd.read_csv(test_dataset_path, index_col='Id')

#Training Target 
y_input = X_full.SalePrice

#Training Predicator
X_input = X_full.drop(columns = 'SalePrice', axis =1)

#Splliting the training data set to TRAINING & VALIDATION data set with 80, 20 ratio.
X_train_full, X_validation_full, y_train, y_validation = train_test_split(X_input, y_input, train_size = 0.8, random_state =0)

#Categorical Columns:
categorical_cols = [col for col in X_train_full.columns
                    if X_train_full[col].nunique()<10 and X_train_full[col].dtype == 'object']
#Numerical Columns:
numerical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in ['int64', 'float64']]

total_valid_cols = categorical_cols + numerical_cols

#Selecting Only valid columns from Training data set
X_train = X_train_full[total_valid_cols]
#Selecting Only valid columns from Validation data set
X_validation = X_validation_full[total_valid_cols]
#Selecting Only valid columns from Ting data set
X_test = X_test_full[total_valid_cols]
print(categorical_cols)
#Imputing the missing values in Numberical Columns:
#I've kept strategy as median, however it is subjective can be MEAN or MOST FREQUENT.
Numerical_transform = SimpleImputer(strategy='median')

#Imputing the missing values & OneHotEncoding in Categorical Columns:

#I've kept strategy as MOST FREQUENT.
# handle_unknown='ignore' : When an unknown category is encountered during testing, it will just ignore that. 
# If we don't set this with ignore code would result into error in an unknown category case.
Categorical_transform = Pipeline(steps= [('cat_impute' ,SimpleImputer(strategy='most_frequent')), 
                                         ('cat_onehot', OneHotEncoder(handle_unknown = 'ignore'))])

#ColumnTransformer constructor takes quite a few arguments:
#The first argument is an array called transformers, which is a list of tuples. The array has the following elements in the same order:
# - name: a name for the column transformer, which will make setting of parameters and searching of the transformer easy.
# - transformer: here we’re supposed to provide an estimator.We’re Imputing the values in Numerical Cols and Imputing & encoding in
#Categorical Columns
# - column(s): the list of columns which you want to be transformed.
     
#Bundle Numerical and Categorical Pre-processing 
column_transform = ColumnTransformer(transformers=[('num', Numerical_transform, numerical_cols),
                                                   ('cat', Categorical_transform, categorical_cols)])
# XGB Model:
model = XGBRegressor(n_estimators=1000,learning_rate = 0.01,random_state=0)

#Bundling preprocessing of columns and MODEL.
final_bundle = Pipeline(steps=[('column_transformation', column_transform), ('model', model)])
final_bundle.fit(X_train, y_train)
y_hat_validation = final_bundle.predict(X_validation)

print("MAE is", mean_absolute_error(y_validation,y_hat_validation))
preds_test = final_bundle.predict(X_test)
output = pd.DataFrame({'ID': X_test.index, 'SalePrice': preds_test}).to_csv('submission_XGB.csv', index=False)