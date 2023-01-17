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
data_train = pd.read_csv('../input/restaurant-revenue-prediction/train.csv.zip',index_col='Id', parse_dates=["Open Date"])
data_test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv.zip',index_col='Id', parse_dates=["Open Date"])
data_train
data_train.describe()
data_train.isnull().sum()
for i in data_train.columns:    
    print(i ,': ',len(data_train[i].unique()))
columnsForDrop = ['Open Date']
data_train.drop(columns=columnsForDrop, inplace=True)
################################
data_test.drop(columns=columnsForDrop, inplace=True)

data_train
s = (data_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = data_train.copy()
label_X_test = data_test.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_encoder.fit(pd.concat([data_train[col], data_test[col]], axis=0, sort=False))
    label_X_train[col] = label_encoder.transform(data_train[col])
    label_X_test[col] = label_encoder.transform(data_test[col])
data_train = label_X_train
data_test = label_X_test
data_train
y = data_train.revenue
############################################
X = data_train.drop(columns=['revenue'])
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error ,explained_variance_score, mean_squared_error
#########################################################################3
from sklearn.ensemble import RandomForestRegressor

parameters = {'max_depth':  list(range(6, 30, 10)),
              'max_leaf_nodes': list(range(50, 500, 100)),
              'n_estimators': list(range(50, 1001, 150))}

parameters1 = {'max_depth':  [6],
              'max_leaf_nodes': [250],
              'n_estimators': [100]}
from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=RandomForestRegressor(),
                       param_grid = parameters, 
                       scoring='neg_mean_squared_error',
                       n_jobs=4,cv=5,verbose=7)

gsearch.fit(X, y)
print(gsearch.best_params_.get('n_estimators'))
print(gsearch.best_params_.get('max_leaf_nodes'))
print(gsearch.best_params_.get('max_depth'))
print(data_train.shape)
print(data_test.shape)
print(X.shape)
final_model = RandomForestRegressor(
                         max_depth = gsearch.best_params_.get('max_depth'),
                           max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes'),
    n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)
final_model.fit(X, y)
preds = final_model.predict(data_test)
print(preds.shape)
print(data_test.shape)
testData = pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip")
submission = pd.DataFrame({
        "Id": testData["Id"],
        "Prediction": preds
    })
submission.to_csv('RandomForestSimple.csv',header=True, index=False)
print('Done')