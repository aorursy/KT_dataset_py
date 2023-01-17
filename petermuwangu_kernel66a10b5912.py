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
data = pd.read_csv("../input/learn-together/train.csv",header=0)

test_data = pd.read_csv('../input/learn-together/test.csv')

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

y = data.Cover_Type

X = data.drop(['Cover_Type'], axis=1)
X_trainf, X_validf, y_train, y_valid = train_test_split(X, y, test_size = 0.8, train_size=0.2, random_state=1)
categorical_cols = [col for col in X_trainf.columns if X_trainf[col].nunique() < 10 and X_trainf[col].dtype =='object']

numerical_cols = [col for col in X_trainf.columns if X_trainf[col].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols

X_train = X_trainf[my_cols].copy()

X_valid = X_validf[my_cols].copy()
numeric_transform = SimpleImputer(strategy='constant')

categorical_transform = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



preprocessor = ColumnTransformer(transformers=[

    ('num', numeric_transform, numerical_cols),

    ('cat', categorical_transform, categorical_cols)

])

model = XGBRegressor(n_estimators=600, learning_rate=0.01, n_jobs=3)



pipe = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])
pipe.fit(X_train, y_train)
predicts = pipe.predict(X_valid)

mae = mean_absolute_error(y_valid, predicts)

print(mae)

print(predicts)
pred_testdata = list(pipe.predict(test_data))

pred_testdata

# pred_testdata.columns = ['Cover_Type']

# data_test = test_data + pred_testdata

# data_test.head()
