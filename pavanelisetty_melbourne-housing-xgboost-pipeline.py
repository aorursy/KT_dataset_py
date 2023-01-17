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
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

data.head()
data=data.drop('Address',axis=1)

data
data.Suburb.nunique()
from sklearn.model_selection import train_test_split

y = data.Price

x = data.drop('Price',axis = 1)

X_train_full , X_valid_full , y_train , y_valid = train_test_split( x , y , random_state=0 , train_size=0.8 , test_size=0.2)

numerical_cols = []

for col in X_train_full.columns:

    if X_train_full[col].dtypes in ['int64','float64']:

        numerical_cols.append(col)

print(numerical_cols)

categorical_cols=[]

for col in X_train_full.columns:

    if X_train_full[col].nunique() < 15 and X_train_full[col].dtype == 'object':

        categorical_cols.append(col)

categorical_cols        
my_cols=numerical_cols + categorical_cols

my_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_train
from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
numerical_transformer = SimpleImputer(strategy = 'constant')
categorical_transformer = Pipeline(steps=[

    ('num',SimpleImputer(strategy='most_frequent')),

('cat',OneHotEncoder(handle_unknown='ignore'))])
#BUNDLE TRANSFORMER



transformer = ColumnTransformer(transformers=[

    ('numb',numerical_transformer,numerical_cols),

    ('category',categorical_transformer,categorical_cols)

])
model = XGBRegressor(n_estimators = 500,

                    learning_rate=0.1)



my_model = Pipeline(steps=[

    ('preprocess',transformer),

    ('model',model)

])
my_model.fit(X_train , y_train)

prediction = my_model.predict(X_valid)

mean_absolute_error(y_valid,prediction)