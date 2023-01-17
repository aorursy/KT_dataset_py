# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loading the data

train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(train.drop('SalePrice',1),train['SalePrice'],test_size=0.2,random_state=0)
missing_cols=[col for col in x_train.columns if x_train[col].isnull().sum()>0]
num_cols=[col for col in missing_cols if (x_train[col].dtype!=object)]

cat_cols=[col for col in missing_cols if (x_train[col].dtype==object)]
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

numerical_transformer=SimpleImputer(strategy='mean')

categorical_transformer=Pipeline(steps=[

    ('imputer',SimpleImputer(strategy='most_frequent')),

    ('onehot',OneHotEncoder(handle_unknown='ignore'))

])
preprocessor=ColumnTransformer(transformers=[

    ('num',numerical_transformer,num_cols),

    ('cat',categorical_transformer,cat_cols)

])
from xgboost import XGBRegressor

model=XGBRegressor(random_state=0)
from sklearn.metrics import mean_absolute_error

my_pipeline=Pipeline(steps=[

    ('preprocessor',preprocessor),

    ('model',model)

])
my_pipeline.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score

scores= -1 * cross_val_score(my_pipeline,x_train,y_train,cv=5,scoring='neg_mean_absolute_error')

print('MAE:\n',scores)
score=mean_absolute_error(y_valid,my_pipeline.predict(x_valid))

print(score)

print('Average MAE:')

print(scores.mean())
my_pipeline.fit(train.drop('SalePrice',1),train['SalePrice'])
pred=my_pipeline.predict(test)
output=pd.DataFrame({'Id':test.Id,

                    'SalePrice':pred})

output.to_csv('submission.csv',index=False)