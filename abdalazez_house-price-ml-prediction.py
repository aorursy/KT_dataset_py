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
data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')

data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

data_train
data_train.describe()
for i in data_train.columns:    

    print(i ,': ',data_train[i].isnull().sum())
cols_with_missing_train = [col for col in data_train.columns

                     if data_train[col].isnull().any()]

cols_with_missing_test = [col for col in data_test.columns

                     if data_test[col].isnull().any()]

#print(cols_with_missing_train)

#print('----------------------')

#print(cols_with_missing_test)

#print(set(cols_with_missing_test) - set(cols_with_missing_train))

all_missing_columns = cols_with_missing_train + cols_with_missing_test

print(len(all_missing_columns))

#Drop columns in training and validation data

data_train.drop(all_missing_columns, axis=1,inplace=True)

data_test.drop(all_missing_columns, axis=1,inplace=True)
# Get  columns whose data type is object i.e. string

filteredColumns = data_train.dtypes[data_train.dtypes == np.object]

# list of columns whose data type is object i.e. string

#print(filteredColumns.index)

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

data_train.drop(listOfColumnNames, axis=1,inplace=True)

data_test.drop(listOfColumnNames, axis=1,inplace=True)
data_train
#for i in data_train.columns:    

 #   print(i ,': ',len(data_train[i].unique()))

#len(data_train.Name.unique)
y = data_train.SalePrice

#############################

X = data_train.drop(columns=['SalePrice'])
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

#explained_variance_score

range_Estimation = getBestScore(1)

minEstim = 1

for i in range(1,100,1):

    #print(getBestScore(i),'*-*',i)

    if range_Estimation > getBestScore(i):

        minEstim = i

print(range_Estimation,'>>>',minEstim)

##### 196 is the best...'''
final_model = XGBRegressor(n_estimators=minEstim,random_state=1,learning_rate=0.05, n_jobs=4)

final_model.fit(X, y)

predictions = final_model.predict(X)

#print(predictions)

#mean_absolute_error(y_true=y , y_pred = predictions)

#print(predictions[:5])

#print(y[:5])
data_test
test_preds = final_model.predict(data_test)

output = pd.DataFrame({'Id': data_test.index,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print('Done')