import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Ridge, Lasso

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import mean_squared_log_error

from lightgbm import LGBMRegressor

from math import sqrt



import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col=[0])

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col=[0])

sample=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv", index_col=[0])
#Dropping features with very high missing values

train_clean=train.drop(columns=['MiscFeature','Fence','PoolQC','FireplaceQu','Alley'])



#Seperating features and Label

X=train_clean.drop(columns=['SalePrice'])

y=train_clean[['SalePrice']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Seperate numerical and categorical features

num_feat=X_train.select_dtypes(include='number').columns.to_list()

cat_feat=X_train.select_dtypes(exclude='number').columns.to_list()



#Pipeline to handle numerical features

num_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])



#Pipeline to handle categorical features

cat_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('encoder', OneHotEncoder(handle_unknown='ignore'))

])



#Transforming numerical and categorical data using pipeline

ct=ColumnTransformer(remainder='drop',

                    transformers=[

                        ('numerical', num_pipe, num_feat),

                        ('categorical', cat_pipe, cat_feat)

                    ])

#Building the model

model=Pipeline([

    ('transformer', ct),   

    ('predictor', LGBMRegressor())

])
model.fit(X_train, y_train);
y_pred_train=model.predict(X_train)

y_pred_test=model.predict(X_test)
print('In sample error: ', sqrt(mean_squared_log_error(y_pred_train, y_train)))

print('Out sample error: ', sqrt(mean_squared_log_error(y_pred_test, y_test)))
#model.fit(X,y);
def submission(test, model):

    y_pred=model.predict(test)

    result=pd.DataFrame(y_pred, index=test.index, columns=['SalePrice'])

    result.to_csv('/kaggle/working/result.csv')
#submission(test, model)