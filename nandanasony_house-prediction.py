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
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor



X = pd.read_csv('/kaggle/input/housing/train.csv')

X.head(1)




X_test = pd.read_csv('/kaggle/input/housing/test.csv')

X_test.head(1)

X = X.drop(columns=['Misc Feature','Fence','Pool QC','Fireplace Qu','Alley'])

X.head(1)





y = X['SalePrice']

X = X.drop(columns=['SalePrice'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
num_cols = X_train.select_dtypes(include='number').columns.to_list()

cat_cols = X_train.select_dtypes(exclude='number').columns.to_list()
num_pipe = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])

cat_pipe = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('encoder', OneHotEncoder(handle_unknown='ignore'))

])

ct = ColumnTransformer(remainder='drop',

                       transformers=[

                           ('numerical', num_pipe, num_cols),

                           ('categorical', cat_pipe, cat_cols)

                       ])

model=Pipeline([

    ('transformer', ct),   

    ('predictor', RandomForestRegressor())

])
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
mean_squared_error(y_pred, y_valid, squared=False)

model.fit(X, y)
y_res = model.predict(X_test)
res = pd.DataFrame({'Id': X_test.PID, 

                    'SalePrice': y_res})

res.to_csv('submission.csv',index=False)
