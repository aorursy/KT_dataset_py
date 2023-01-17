# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data test

train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
# read data test

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test.head()
# chose feature and target

x = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'LotShape', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt']

y = ['SalePrice']
# build pipeline

numerical_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt']

categorical_cols = [ 'MSZoning', 'LotShape', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle']

# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



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
# build model

model = RandomForestRegressor(n_estimators=100, random_state=0)
engine = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

# train model

engine.fit(train[x], train[y])



# prediction result

result = engine.predict(test[x])
# submit

submission = pd.DataFrame({

        "Id": test['Id'],

        "SalePrice": result

    })

# save

submission.to_csv('submission.csv', index=False)

submission.head()