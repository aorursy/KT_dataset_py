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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
dataset = train.append(test)
print(type(dataset))
dataset.info()
dataset = dataset.drop('Id', axis =1)
float_columns = [i for i in dataset.columns  if dataset[i].dtypes == float and dataset[i].isnull().sum() > 0]
print(float_columns)
obj_columns = [i for i in dataset.columns  if dataset[i].dtypes == object and dataset[i].isnull().sum() > 0]
print(obj_columns)
for column in float_columns:
   dataset[column] = dataset[column].replace(np.nan,0.0)    
for column in obj_columns:
    dataset[column] = dataset[column].replace(np.nan,'None')

    
required_columns = [i for i in dataset.columns if dataset[i].dtypes==object]
   
        
print(required_columns)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def label_encoding(col):
    col = le.fit_transform(col)
    return col
for i in required_columns:
    dataset[i] = label_encoding(dataset[i])
print(dataset.info())
X_train = dataset.iloc[:1460, :-1].values
y_train = dataset.iloc[:1460, -1].values
X_test = dataset.iloc[1460: ,:-1].values
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, scoring = 'r2')
print(scores)
y_pred = regressor.predict(X_test)
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
sample_submission['SalePrice'] = y_pred
sample_submission.head(10)
sample_submission.to_csv('submission.csv', index=False)