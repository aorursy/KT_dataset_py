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
!cat /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data['SalePrice']          

X = train_data.drop(['SalePrice'], axis=1)



X.columns
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]

my_cols = numerical_cols + categorical_cols
X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test_data[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from math import sqrt



numerical_transformer = Pipeline(steps=[

    ('imputer',SimpleImputer(strategy='mean'))

])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



from sklearn.linear_model import LinearRegression



model = LinearRegression()



clf = Pipeline(steps=[('preprocessor', preprocessor), 

                      ('model', model)])



clf.fit(X_train, y_train)

preds = clf.predict(X_valid)

mean_squared_error(preds, y_valid)
from xgboost import XGBRegressor



def check(n = 500, r = 0.01):





    model = XGBRegressor(n_estimators=n, learning_rate=r)



    clf = Pipeline(steps=[('preprocessor', preprocessor), 

                      ('model', model)])



    clf.fit(X_train, y_train)

    preds = clf.predict(X_valid)

    return sqrt(mean_squared_error(y_valid, preds))







#for n in range(500, 1201, 100):

#    print(str(n), str(check(n=n, r=0.05)))
check(n=1100, r=0.05)
model = XGBRegressor(n_estimators=1100, learning_rate=0.05)



clf = Pipeline(steps=[('preprocessor', preprocessor), 

                      ('model', model)])

clf.fit(X_train, y_train)



preds_test = clf.predict(X_test)



output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)