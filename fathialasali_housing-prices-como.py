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
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

train_data
train_data.info()
train_data.describe()
cols_with_missing_train = [col for col in train_data.columns

                     if train_data[col].isnull().any()]

print(cols_with_missing_train)

print('----------------------')

print(set(cols_with_missing_train))

train_data.drop(cols_with_missing_train, axis=1,inplace=True)



filteredColumns = train_data.dtypes[train_data.dtypes == np.object]



print(filteredColumns.index)

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

train_data.drop(listOfColumnNames, axis=1,inplace=True)

train_data.drop(columns=['BsmtUnfSF', 'BsmtFinSF1','GarageCars', 'GarageArea','BsmtFinSF2','BsmtHalfBath','TotalBsmtSF','BsmtFullBath'], inplace=True)
train_data.dtypes
train_data
y = train_data.SalePrice

X = train_data.drop(columns=['SalePrice'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error ,explained_variance_score, mean_squared_error

def getBestScore(n_est):

    my_model = XGBRegressor(n_estimators=n_est,random_state=1,learning_rate=0.05, n_jobs=4)

    my_model.fit(X_train, y_train)

    predictions = my_model.predict(X_test)

    mean_Error = mean_squared_error(y_true=y_test,y_pred = predictions)

    return mean_Error 


range_Estimation = getBestScore(1)

minEstim = 1

for i in range(1,100,1):



    if range_Estimation > getBestScore(i):

        minEstim = i

print(range_Estimation,'>>>',minEstim)

final_model = XGBRegressor(n_estimators=minEstim,random_state=1,learning_rate=0.05, n_jobs=4)

final_model.fit(X, y)

predictions = final_model.predict(X)

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')

test_data
cols_with_missing_test = [col for col in test_data.columns

                     if test_data[col].isnull().any()]

print(cols_with_missing_test)

print('----------------------')

print(len(cols_with_missing_test))

test_data.drop(cols_with_missing_test, axis=1,inplace=True)
filteredColumns = test_data.dtypes[test_data.dtypes == np.object]

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

test_data.drop(listOfColumnNames, axis=1,inplace=True)
test_data.dtypes
test_data
final_model = XGBRegressor(n_estimators=minEstim,random_state=1,learning_rate=0.05, n_jobs=4)

final_model.fit(X, y)

predictions = final_model.predict(X)
test_preds = final_model.predict(test_data)

output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print('Done')