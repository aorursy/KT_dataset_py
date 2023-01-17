# Code you have previously used to load data

import numpy as np

import pandas as pd



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Ridge, Lasso

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor





from math import sqrt

# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

test_data_path = '../input/home-data-for-ml-course/test.csv'

test_data = pd.read_csv(test_data_path)

sample_path = '../input/home-data-for-ml-course/sample_submission.csv'

sample = pd.read_csv(sample_path)
train_clean=home_data.drop(columns=['MiscFeature','Fence','PoolQC','FireplaceQu','Alley'])

X=train_clean.drop(columns=['SalePrice'])

y=home_data[['SalePrice']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.15, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
num_feat=X_train.select_dtypes(include='number').columns.tolist()

cat_feat=X_train.select_dtypes(exclude='number').columns.tolist()
num_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])

cat_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('encoder', OneHotEncoder(handle_unknown='ignore'))

])

ct=ColumnTransformer(remainder='drop',

                    transformers=[

                        ('numerical', num_pipe, num_feat),

                        ('categorical', cat_pipe, cat_feat)

                    ])

model=Pipeline([

    ('transformer', ct),   

    ('predictor', GradientBoostingRegressor())

])
model.fit(X_train, y_train);
y_pred_train=model.predict(X_train)

y_pred_test=model.predict(X_test)
print('Train mean absolute error: ', round(mean_absolute_error(y_pred_train, y_train)))

print('Test mean absolute error: ', round(mean_absolute_error(y_pred_test, y_test)))
def submission(test, model):

    y_pred=model.predict(test)

    result=pd.DataFrame({'Id':sample.Id, 'SalePrice':y_pred})

    result.to_csv('/kaggle/working/result.csv',index=False)

submission(test_data, model)
check=pd.read_csv('/kaggle/working/result.csv')

check.head()
import os

os.chdir(r'../working')

from IPython.display import FileLink

FileLink(r'result.csv')