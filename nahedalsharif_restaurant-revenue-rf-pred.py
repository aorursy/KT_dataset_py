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
#load data
train = pd.read_csv('../input/restaurant-revenue-prediction/train.csv.zip',index_col='Id')
test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv.zip',index_col='Id')
train
train.columns
train['Open_Date']=train['Open Date']
train.drop('Open Date',axis=1,inplace=True)

train['City_Group']=train['City Group']
train.drop('City Group',axis=1,inplace=True)

#----------------

test['Open_Date']=test['Open Date']
test.drop('Open Date',axis=1,inplace=True)

test['City_Group']=test['City Group']
test.drop('City Group',axis=1,inplace=True)
train.columns
train.isnull().sum()
train.dtypes

train['Year'] = pd.DatetimeIndex(train['Open_Date']).year
train['Month'] = pd.DatetimeIndex(train['Open_Date']).month
train['Day'] = pd.DatetimeIndex(train['Open_Date']).day
train.drop('Open_Date',axis=1,inplace=True)

test['Year'] = pd.DatetimeIndex(test['Open_Date']).year
test['Month'] = pd.DatetimeIndex(test['Open_Date']).month
test['Day'] = pd.DatetimeIndex(test['Open_Date']).day
test.drop('Open_Date',axis=1,inplace=True)
train.head()
train.describe()
y_train = train.revenue

train.drop('revenue',axis=1,inplace=True)
obj_cols= train.columns[train.dtypes == 'object']
obj_cols
from sklearn.preprocessing import LabelEncoder


#le_train =train.copy()
#le_test =test.copy()

# Apply label encoder to each column with categorical data
le = LabelEncoder()
for col in obj_cols:
    le.fit(pd.concat([train[col], test[col]], axis=0, sort=False))
   
    train[col] = le.transform(train[col])
    
    test[col] = le.transform(test[col])
train
my_randome_state=1486

parameters = {
    'n_estimators': list(range(100, 1001, 100)), 
    'max_leaf_nodes': list(range(2, 50, 5)), 
    'max_depth': list(range(6, 70, 5))
}
parameters
from sklearn.ensemble import RandomForestRegressor


#parameter1 = {'max_depth':  [6],
#              'max_leaf_nodes': [250],
#              'n_estimators': [100]}




#parameters = {'max_depth':  list(range(6, 30, 10)),
#              'max_leaf_nodes': list(range(50, 500, 100)),
#             'n_estimators': list(range(50, 1001, 150))}


from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=RandomForestRegressor(),
                       param_grid = parameters, 
                       scoring='neg_mean_squared_error',
                       n_jobs=4,cv=5,verbose=7)

gsearch.fit(train, y_train)
print(gsearch.best_params_.get('n_estimators'))
print(gsearch.best_params_.get('max_leaf_nodes'))
print(gsearch.best_params_.get('max_depth'))
final_model = RandomForestRegressor(
                         max_depth = gsearch.best_params_.get('max_depth'),
                           max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes'),
    n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)
final_model.fit(train, y_train)
test
preds = final_model.predict(test)
len(preds)
# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': preds})
output
output.to_csv('submission.csv', index=False)
print('done!')