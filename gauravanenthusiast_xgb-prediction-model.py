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
X_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X_train.describe()
to_drop_cols = [col for col in X_train.columns if X_train[col].isnull().sum() > 500]

print(to_drop_cols)
X_train.drop(to_drop_cols, axis=1, inplace = True)

y_train = X_train['SalePrice']

X_train.drop(['SalePrice'], axis=1, inplace=True)

X_test.drop(to_drop_cols, axis=1, inplace=True)
X_train.info()
missing_data_cols = [col for col in X_train.columns if X_train[col].isnull().any()]

print(missing_data_cols)



missing_test_cols =[col for col in X_test.columns if X_test[col].isnull().any()]

print(missing_test_cols)
X_train[missing_data_cols].describe()
object_cols = [col for col in X_train.columns if X_train[col].dtypes=='object']

print(len(object_cols))
numerical_cols = [col for col in X_train.columns if X_train[col].dtypes in ['float64', 'int64']]

#numerical_cols = list(numericals[numericals].index)

print(len(numerical_cols))
cols = object_cols + numerical_cols

X_train = X_train[cols]

X_test = X_test[cols]
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



numerical_transformer = SimpleImputer(strategy = 'constant')



categorical_transformer = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

])



preprocessor = ColumnTransformer(transformers = [

    ('num', numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, object_cols)

])
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score



my_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, max_depth= 7)



my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', my_model)

])



my_pipeline.fit(X_train, y_train)



scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE scores:\n", scores)
preds = my_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)