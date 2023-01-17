# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import CSV files

X_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')

X_test_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col = 'Id')

X_full.head()
#Group columns

y = X_full['SalePrice']

X_full.drop(['SalePrice'], axis=1, inplace=True)

numeric_cols = [cols for cols in X_full.columns if X_full[cols].dtype in ['int64','float64']]

categoric_cols = [cols for cols in X_full.columns if X_full[cols].dtype == 'object']

low_car_cols = [cols for cols in X_full.columns if X_full[cols].nunique() < 8 and X_full[cols].dtype == 'object']



used_cols = numeric_cols + low_car_cols

X = X_full[used_cols].copy()

X_test = X_test_full[used_cols].copy()

X.head()
dropped_cols = [cols for cols in X.columns if X[cols].isnull().sum() > 250]

X.drop(dropped_cols, axis=1, inplace=True)

X_test.drop(dropped_cols, axis=1, inplace=True)
num_cols = [cols for cols in X.columns if X[cols].dtype in ['int64','float64']]

cat_cols = [cols for cols in X.columns if X[cols].dtype in ['object']]
my_imputer = SimpleImputer(strategy = 'median')

cat_imputer = SimpleImputer(strategy='most_frequent')

ohe = OneHotEncoder(handle_unknown = 'ignore', sparse=False)



X_num = X[num_cols].copy()

X_cat = X[cat_cols].copy()



imputed_X = pd.DataFrame(my_imputer.fit_transform(X_num))



imputed_X_cat = pd.DataFrame(cat_imputer.fit_transform(X_cat))

X_ohe_cat = pd.get_dummies(X_cat)

X_ohe_cat.reset_index(inplace=True)



X_final = pd.concat([imputed_X,X_ohe_cat], axis=1)



removed_cols = ['Utilities_NoSeWa','Heating_Floor','Heating_OthW','Electrical_Mix','GarageQual_Ex']



for cols in removed_cols:

    X_final.drop(cols, axis=1, inplace = True)

    X_final.head()
X_train,X_valid,y_train,y_valid = train_test_split(X_final,y,random_state = 1)
model = XGBRegressor(n_estimators = 500, learning_rate=0.1)

model.fit(X_train,y_train,early_stopping_rounds = 5, eval_set=[(X_valid,y_valid)], verbose=False)
X_num_test = X_test[num_cols].copy()

X_cat_test = X_test[cat_cols].copy()



imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_num_test))



imputed_X_cat_test = pd.DataFrame(cat_imputer.fit_transform(X_cat_test))

X_ohe_cat_test = pd.get_dummies(X_cat_test)

X_ohe_cat_test.reset_index(inplace=True)



X_final_test = pd.concat([imputed_X_test,X_ohe_cat_test], axis=1)
preds = model.predict(X_final_test)
my_submission = pd.DataFrame({'Id': X_final_test.Id, 'SalePrice': preds})

my_submission.to_csv('submission.csv', index=False)