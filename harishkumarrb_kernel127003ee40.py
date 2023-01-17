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
train_data  = pd.read_csv('../input/titanic/train.csv')
train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")
test_data.head()
num_missing = [col for col in train_data.columns if train_data[col].isnull().any()]
num_missing
col_missing = [col for col in train_data.columns if train_data[col].dtypes=='object']
col_missing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#numerical preprocessing
numerical_transformer = SimpleImputer(strategy='constant')

#categorilcal preprocessing
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                            ])
#bundle preprocessing
preprocessor = ColumnTransformer(transformers = [('num', numerical_transformer,num_missing),
                                              ('col',categorical_transformer,col_missing)])
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=100, learning_rate=0.05)

my_pipeline = Pipeline(steps = [('preprocessor',preprocessor),
                               ('my_model',my_model)])


