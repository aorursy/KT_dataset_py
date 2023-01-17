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
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,auc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')

train = train[train.GrLivArea < 4500]
train.reset_index(drop=True,inplace=True)
label = train[['SalePrice']]
train.drop('SalePrice',axis=1,inplace=True)
train.head()
label.head()
train.info()
train.describe()
numerical_col = []
category_col = []
for x in train.columns:
    if train[x].dtype == 'object':
        category_col.append(x)
        print(x+': ' + str(len(train[x].unique())))
    else:
        numerical_col.append(x)
        
print('CATEGORY column \n', category_col)
print('Numerical column\n',numerical_col)
numerical_col.remove('MSSubClass')
category_col.append('MSSubClass')
print(category_col)
train_numerical = train[numerical_col]
train_numerical.head()
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(train_numerical)
train_numerical = imputer.transform(train_numerical)
