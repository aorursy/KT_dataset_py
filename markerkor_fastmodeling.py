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
train.iloc[:5,:20]
train.iloc[:,:20].dtypes
train.iloc[:5,20:40]
train.iloc[:,20:40].dtypes
train.iloc[:5,40:60]
train.iloc[:,40:60].dtypes
train.iloc[:5,60:80]
train.iloc[:,60:80].dtypes
train.columns
for col in train.columns:
    if not col in test.columns:
        print(col)
for col in test.columns:
    if not col in train.columns:
        print(col)
from sklearn.model_selection import train_test_split
train.drop(['SalePrice'],axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['SalePrice'],axis=1), train['SalePrice'], test_size=0.2, random_state=2020, shuffle=True)
cat_features = [col for col in X_train.columns if X_train[col].dtype=='object']
num_features = [col for col in X_train.columns if X_train[col].dtype in ['int64','float64']]
len(cat_features), len(num_features), len(X_train.columns)
for col in cat_features:
    val = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(val)
    X_valid[col] = X_valid[col].fillna(val)
    X_test[col] = X_test[col].fillna(val)
    
for col in num_features:
    val = X_train[col].mean()
    X_train[col] = X_train[col].fillna(val)
    X_valid[col] = X_valid[col].fillna(val)
    X_test[col] = X_test[col].fillna(val)
X_train.isnull().sum().sort_values(ascending=False)
y_train.isnull().sum()
from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in cat_features:
    encoder = LabelEncoder()
    X_ = pd.concat([X_train[col],X_valid[col]],axis=0)
    encoder.fit(X_)
    X_train[col] = encoder.transform(X_train[col])
    X_valid[col] = encoder.transform(X_valid[col])
    encoders[col] = encoder
for col in cat_features:
    encoder = encoders[col]
    encoder.transform(X_test[col])
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(y,preds):
    return sqrt(mean_squared_error(y, preds))
y_train = np.log(y_train)
y_valid = np.log(y_valid)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
preds = rf.predict(X_valid)
rmse(y_valid,preds)
tmp = pd.DataFrame({'importance':rf.feature_importances_,'feature':X_train.columns}).sort_values('importance', ascending=False)
tmp.iloc[:30]