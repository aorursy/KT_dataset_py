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
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
train.describe()
train
y = train.SalePrice

y.describe()
X = train.drop(['Id', 'SalePrice'], axis=1)
X.describe()
X
from sklearn.model_selection import train_test_split

val_X, train_X, val_y, train_y = train_test_split(X,y, random_state=0, train_size=0.8, test_size=0.2)
val_X
train_X
s = (train.dtypes == 'object')

object_cols = list(s[s].index)

print('categorical data')

print(object_cols)
categorical_cols = [cname for cname in train_X if train_X[cname].nunique() <=5 and train_X[cname].dtype == 'object']
numerical_cols = [cname for cname in train_X.columns if train_X[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols

X_train = train_X[my_cols].copy()

X_val = val_X[my_cols].copy()
X_train
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



numerical_transformer = SimpleImputer(strategy= 'constant')

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state=0, n_estimators = 200)
from sklearn.metrics import mean_absolute_error



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



my_pipeline.fit(X_train, train_y)



preds = my_pipeline.predict(X_val)

print('Mean Absolute Error:', mean_absolute_error(preds, val_y))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def get_mae(max_leaf_nodes, X_train, X_val, train_y, val_y):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0, n_estimators=200)

    my_pipeline.fit(X_train, train_y)

    preds = my_pipeline.predict(X_val)

    mae = mean_absolute_error(preds, val_y)

    return (mae)
for max_leaf_nodes in [5, 35, 50, 65, 75, 100, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, X_train, X_val, train_y, val_y)

    print('mae: ',  max_leaf_nodes, my_mae)
# Test Data
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
test.describe()
test.head()
test
X_test = test.drop(['Id'], axis=1)
X_test
X_test.equals(X)
predicted_prizes = my_pipeline.predict(X_test)
predicted_prizes
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prizes})

my_submission.to_csv('submission.csv', index=False)