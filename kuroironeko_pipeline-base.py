# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# Separate target from predictors

y = df.SalePrice

X = df.drop(['SalePrice', 'Id'], axis=1)



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train.info()
categorical_cols = X_train.select_dtypes(include = 'object').columns

numerical_cols = X_train.select_dtypes(exclude = 'object').columns
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from lightgbm import LGBMRegressor



model = LGBMRegressor(random_state=0)
from sklearn.metrics import mean_absolute_error



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = np.sqrt(mean_absolute_error(y_valid, preds))

print('RMSE:', score)