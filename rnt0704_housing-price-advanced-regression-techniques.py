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


#importing various libraries to use

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



#read the data

data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#separate train from target

y = data_train.SalePrice

X = data_train.drop(['SalePrice'], axis = 1)



#divide data into train test split

X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(

                                                        X,y,test_size = 0.2, random_state = 0)



#select numerical cols

numerical_cols = [cname for cname in X_train_full.columns if

                 X_train_full[cname].dtype in ['int64','float64']]



#selecting the low cardinality values

categorical_cols = [cname for cname in X_train_full.columns if

                   X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == 'object']



my_cols = numerical_cols + categorical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_train.head()

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



#preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



#preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer',SimpleImputer(strategy='most_frequent')),

    ('onehot',OneHotEncoder(handle_unknown = 'ignore'))

])



#Bundle preprocessing for numerical data and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num',numerical_transformer,numerical_cols),

        ('cat',categorical_transformer,categorical_cols)

    ])
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators = 250, random_state = 0)



from sklearn.metrics import mean_absolute_error



#bundle preprocessing and model code

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),

                             ('model',model)])



#fit the model

my_pipeline.fit(X_train,y_train_full)



#predicting validation data

preds_val = my_pipeline.predict(X_valid)



#score error

score = mean_absolute_error(y_valid_full,preds_val)

print('MAE:',score)
#preprocessing test data and fit

final_pred = my_pipeline.predict(data_test)

#saving to the output

output = pd.DataFrame({'Id':data_test.Id,

                     'SalePrice':final_pred})

output.to_csv('submission.csv',index=False)