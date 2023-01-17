# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import skew
from scipy.stats.stats import pearsonr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
all_data = pd.concat([train.loc[:, 'MSSubClass': 'SaleCondition'], 
                      test.loc[:, 'MSSubClass': 'SaleCondition']], axis=0)
numeric_features = all_data[all_data.dtypes[all_data.dtypes != 'O'].index]
discrete_features = all_data[all_data.dtypes[all_data.dtypes == 'O'].index]
all_data.head()
# # drop nan
# train.drop(['Id', 'Alley', 'PoolQC', 'Fence'], axis=1, inplace=True)
price = pd.DataFrame({'price': train.SalePrice, 'log(price + 1)': np.log1p(train.SalePrice)})
price.hist()
train.SalePrice = np.log1p(train.SalePrice)
numeric_features = all_data.dtypes[all_data.dtypes != 'O'].index
discrete_features = all_data.dtypes[all_data.dtypes == 'O'].index
skewed_feats = train[numeric_features].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
lr = LinearRegression(normalize=True)
lr.fit(X_train, y)
lr.score(X_train, y)
yhat = lr.predict(X_test)

features = pd.concat([numeric_features, discrete_features], axis=1)
# y = features['SalePrice']
# features.drop(['SalePrice'], axis=1, inplace=True)
print(features.shape)
# print(y.shape)
features_normalize = scale(features)
y = features_normalize[:, 36]
X = np.hstack([features_normalize[:, :36] , features_normalize[:, 37:]])
print(X.shape)
pca = PCA(n_components=0.9)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)
lr = LinearRegression(normalize=True)
lr.fit(X, y)
# drop nan
test.drop(['Id', 'Alley', 'PoolQC', 'Fence'], axis=1, inplace=True)
numeric_features_test = test[test.dtypes[test.dtypes != 'O'].index]
discrete_features_test = test[test.dtypes[test.dtypes == 'O'].index]
numeric_features_test = numeric_features_test.fillna(numeric_features_test.mean())
discrete_features_test = pd.get_dummies(discrete_features_test)
print(numeric_features_test.shape)
print(discrete_features_test.shape)
features_test = pd.concat([numeric_features_test, discrete_features_test], axis=1)
print(features_test.shape)
print(lr.score(X, y))
print(X.shape)
print(features_test.shape)
