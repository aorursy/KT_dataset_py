from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
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
train=pd.read_csv('/kaggle/input/allstate-claims-severity/train.csv')

test=pd.read_csv('/kaggle/input/allstate-claims-severity/test.csv')

sample_submission=pd.read_csv('/kaggle/input/allstate-claims-severity/sample_submission.csv')
X=train.drop(columns='loss')

y=train.loss
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
num_feat=X_train.select_dtypes(include='number').columns.to_list()

cat_feat=X_train.select_dtypes(exclude='number').columns.to_list()
num_pipe=Pipeline([

    ('impute',SimpleImputer(strategy='median')),

    ('scale',StandardScaler())

])



cat_pipe= Pipeline([

    ('impute',SimpleImputer(strategy='most_frequent')),

    ('encode',OneHotEncoder(handle_unknown='ignore'))

])



ct=ColumnTransformer(remainder='drop',

                    transformers=[

                        ('num',num_pipe, num_feat),

                        ('cat', cat_pipe, cat_feat)]

                    )

model=Pipeline([

    ('transformer', ct),

    ('predictor', AdaBoostRegressor())

])
model.fit(X_train, y_train);
print('In Sample Score: ', model.score(X_train, y_train))

print('Out Sample Score: ', model.score(X_test, y_test))
y_pred=model.predict(test)
submission=pd.DataFrame({'id':sample_submission.id, 'loss': y_pred})

submission.to_csv('/kaggle/working/submission.csv', index=False)

result=pd.read_csv('/kaggle/working/submission.csv')

result.head()