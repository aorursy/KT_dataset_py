import pandas as pd

import numpy as np 

from sklearn import linear_model,cross_validation

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

%matplotlib inline 



train = pd.read_csv('/kaggle/input/train.csv')

test = pd.read_csv('/kaggle/input/test.csv')
train.columns.unique()
train_label = train.pop('SalePrice')

features = pd.concat([train,test], keys = ['train', 'test'] )
out = pd.concat([train.isnull().sum(), test.isnull().sum()],axis = 1 ,keys = ['train','test'])

out[out.train > 1000]
useless_data = ['Alley','FireplaceQu','PoolQC','Fence','Street','MiscFeature','GarageArea','BsmtHalfBath','BsmtFullBath','GarageYrBlt','3SsnPorch','Utilities','MasVnrType','MasVnrArea','BsmtFinSF2','BsmtFinSF1','PavedDrive','BsmtUnfSF']

features.drop(useless_data, inplace= True, axis = 1)
features['MSZoning'].fillna(features['MSZoning'].mode(), inplace = True)

features['LotFrontage'].fillna(0, inplace = True)

features['GarageCond'].fillna(0)

features['LotShape'].fillna(features['LotShape'].mode(), inplace = True)

for items in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):

    features[items].fillna('NoBSMT', inplace = True)

# features['BsmtQual'].fillna('NoBsmt')

# features['BsmtCond'].fillna(0)

# features['BsmtExposure'].fillna(0)

# features['Bsmt'].fillna(0)

features['Electrical'].fillna(features['Electrical'].mode(), inplace = True)

features['KitchenQual'].fillna(features['KitchenQual'].mode(), inplace = True)

features['SaleType'].fillna(features['SaleType'].mode(), inplace = True)

features['SaleType'].fillna(features['SaleType'].mode(), inplace = True)

for items in ('GarageType', 'GarageFinish', 'GarageQual','GarageCond'):

    features[items].fillna('NoGrg', inplace = True)

features['GarageCars'].fillna(0, inplace = True)

features['TotalBsmtSF'].fillna(0, inplace = True)

features['Exterior1st'].fillna(features['Exterior1st'].mode(), inplace =True)
for items in ('LotArea','1stFlrSF','2ndFlrSF', 'TotRmsAbvGrd'):

    features[items] = np.log1p(features[items])

train_label = np.log1p(train_label)
items = features.columns[features.dtypes == 'O']
features = pd.get_dummies(features, columns= items)
train_features = features[:train.shape[0]]

test_features = features[train.shape[0]:]
x_train,x_test,y_train,y_test = cross_validation.train_test_split(train_features,train_label, test_size = 0.1)
clf = linear_model.LinearRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
len(clf.predict(test_features))
Sales_prediction = np.exp(clf.fit(train_features, train_label).predict(test_features))
pd.DataFrame({'Id': test.Id, 'SalePrice': Sales_prediction}).to_csv('prediction_file.csv', index =False)