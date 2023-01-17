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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')
train.columns

test.columns
#new_train=train.drop(columns='SalePrice')
train.columns

for i in train.columns:    

    print(i ,': ',train[i].isnull().sum())
cols_with_missing_train = [col for col in train.columns

                     if train[col].isnull().any()]

cols_with_missing_test = [col for col in test.columns

                     if test[col].isnull().any()]

all_missing_columns = cols_with_missing_train + cols_with_missing_test

print(len(all_missing_columns))

train.drop(all_missing_columns, axis=1,inplace=True)

test.drop(all_missing_columns, axis=1,inplace=True)
filteredColumns =train.dtypes[train.dtypes == np.object]

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

train.drop(listOfColumnNames, axis=1,inplace=True)

test.drop(listOfColumnNames, axis=1,inplace=True)
y = train.SalePrice

X = train.drop(columns=['SalePrice'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error ,explained_variance_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

parameters = {'max_depth':  [6, 10, 15],

              'max_leaf_nodes': [30,50,100],

              'n_estimators': [400,500]}



from sklearn.model_selection import GridSearchCV



gsearch = GridSearchCV(estimator=RandomForestRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5,verbose=7)



gsearch.fit(X_train, y_train)
print (gsearch.best_params_.get('n_estimators'))

print(gsearch.best_params_.get('max_depth'))
my_model = RandomForestRegressor(

                         max_depth = gsearch.best_params_.get('max_depth'),

                           max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes'),

              n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)

my_model.fit(X_train, y_train)

predictions = my_model.predict(X_test)

mean_Error = mean_absolute_error(y_true=y_test,y_pred = predictions)

print(mean_Error)
def getBestScore(n_est):

    my_model = RandomForestRegressor(n_estimators=n_est,random_state=1,learning_rate=0.05, n_jobs=4)

    my_model.fit(X_train, y_train)

    predictions = my_model.predict(X_test)

    mean_Error = mean_absolute_error(y_true=y_test,y_pred = predictions)

    return mean_Error 
final_model = RandomForestRegressor(

                         max_depth = gsearch.best_params_.get('max_depth'),

                           max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes'),

              n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)



final_model.fit(X, y)

predictions = final_model.predict(X)
predictions

test_preds = final_model.predict(test)

output = pd.DataFrame({'Id': test.index,

                       'SalePrice': test_preds})

output.to_csv('submission3.csv', index=False)

print('Done')