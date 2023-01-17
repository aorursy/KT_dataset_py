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
df = pd.read_csv('../input/test-dataset/dataset_00_with_header.csv');

df
df.describe()


cols_with_missing = [col for col in df.columns if df[col].isnull().any()]



print(cols_with_missing)
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer(strategy='most_frequent')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(df))

# imputed_X_valid = pd.DataFrame(my_imputer.transform(data_test))



# Imputation removed column names; put them back

imputed_X_train.columns = df.columns

# imputed_X_valid.columns = data_test.columns



df = imputed_X_train

df
cols_with_missing = [col for col in df.columns if df[col].isnull().any()]

cols_with_missing
string_Data = (df.dtypes == 'object')

object_cols = list(string_Data[string_Data].index)



print("Categorical variables:")

print(object_cols)
y = df.y

#############################

X = df.drop(columns=['y'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error ,explained_variance_score, mean_squared_error
parameters1 = {'learning_rate':  [x / 100 for x in range(5, 101, 5)],

              'max_depth':  list(range(6, 30, 6)),

              'n_estimators': list(range(50, 1001, 50))}
parameters = {'learning_rate':  [0.02],

              'max_depth':  [6],

              'n_estimators': [100]}
# from sklearn.model_selection import GridSearchCV



# gsearch = GridSearchCV(estimator=XGBRegressor(),

#                        param_grid = parameters, 

#                        scoring='neg_mean_squared_error',

#                        n_jobs=4,cv=5,verbose=7)



# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_.get('n_estimators'))

# print(gsearch.best_params_.get('learning_rate'))

# print(gsearch.best_params_.get('max_depth'))
# my_model = XGBRegressor(learning_rate = gsearch.best_params_.get('learning_rate'),

#                          max_depth = gsearch.best_params_.get('max_depth'),

#               n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)



# my_model.fit(X_train, y_train)



# predictions = my_model.predict(X_test)



# squared_error = mean_squared_error(y_true=y_test,y_pred = predictions)



# print(squared_error)
my_model = XGBRegressor(learning_rate =0.2,

                         max_depth = 6,

              n_estimators = 70,random_state=1, n_jobs=4)



my_model.fit(X_train, y_train)



predictions = my_model.predict(X_test)



squared_error = mean_squared_error(y_true=y_test,y_pred = predictions)



print(squared_error)