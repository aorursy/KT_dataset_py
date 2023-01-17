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
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')



train_data
train_data.info()
train_data.isnull().sum()
cols_with_miss_val_train = [col for col in train_data.columns

                     if train_data[col].isnull().any()]

cols_with_miss_val_test = [col for col in test_data.columns

                     if test_data[col].isnull().any()]



miss_val_columns = cols_with_miss_val_train + cols_with_miss_val_test

print(miss_val_columns)

print('------------')

print(len(miss_val_columns))

diff_col= set(test_data)-set(train_data)

train_data.drop(columns=diff_col,axis = 1, inplace=True)

test_data.drop(columns=diff_col,axis = 1, inplace=True)
cols_with_miss_train_data = [col for col in train_data.columns 

                            if train_data[col].isnull().any()]



cols_with_miss_test_data = [col for col in test_data.columns 

                            if test_data[col].isnull().any()]



all_missing_columns = cols_with_miss_train_data + cols_with_miss_test_data

train_data.drop(columns=all_missing_columns, axis=1,inplace=True)

test_data.drop(columns=all_missing_columns, axis=1,inplace=True)
filteredColumns = train_data.dtypes[train_data.dtypes == np.object]

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

train_data.drop(listOfColumnNames, axis=1,inplace=True)

test_data.drop(listOfColumnNames, axis=1,inplace=True)
target_col = 'SalePrice'

y = train_data[target_col]

X = train_data.select_dtypes(exclude='object')

X = X.drop(columns=[target_col])

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from xgboost.sklearn import XGBRegressor

parameters = [{

'n_estimators': list(range(100, 501, 100)), 

'learning_rate': [0.1,0.2,0.3],

    'max_depth':[6,7,8]

}]

from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5)



gsearch.fit(X_train, y_train)



gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate'),gsearch.best_params_.get('max_depth')
f_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),learning_rate = gsearch.best_params_.get('learning_rate'),

                       max_depth =gsearch.best_params_.get('max_depth'),random_state = 1 )

f_model.fit(X_train, y_train)

pred = f_model.predict(X_train)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_train,pred))

    
fin_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),learning_rate = gsearch.best_params_.get('learning_rate'),

                       max_depth =gsearch.best_params_.get('max_depth'),random_state = 1 )

fin_model.fit(X, y)

pred = fin_model.predict(test_data)
test_out = pd.DataFrame({

    'Id': test_data.index, 

    'SalePrice': pred,

})
test_out.to_csv('submission.csv', index=False)