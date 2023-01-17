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
from matplotlib import pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.shape
train_nulls = train.isnull().sum().sort_values(ascending=False) / train.shape[0]
null_columns = train_nulls[train_nulls > 0.3].index
train = train.drop(null_columns, axis=1)
train.shape
target = train['SalePrice']
train = train.drop('Id', axis=1)
train.shape
corr_matrix.shape
corr_matrix = train.drop('SalePrice', axis=1).corr()
fig = plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix)
corr_thresh = corr_matrix[corr_matrix.abs() > 0.5]
big_correlation_features = corr_thresh.sum()[corr_thresh.sum() > 1].index
train = train.drop(big_correlation_features, axis=1)
train.head()
train = train.drop('SalePrice', axis=1)
cat_features = train.select_dtypes(include=object).columns
num_features = train.select_dtypes(exclude=object).columns
assert len(cat_features) + len(num_features) == train.shape[1]
len(num_features), len(cat_features)
train[num_features].isnull().sum().sort_values(ascending=False)
train['MasVnrArea'].hist()
train['LotFrontage'].hist()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[num_features], target, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.log(y_train)
y_test = np.log(y_test)
from sklearn.linear_model import Ridge
lr = Ridge(alpha=1)
lr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
predicts = lr.predict(X_test)
loss = np.sqrt(mean_squared_error(y_test, predicts))
loss
