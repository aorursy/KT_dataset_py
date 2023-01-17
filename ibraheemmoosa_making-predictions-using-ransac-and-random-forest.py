# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

sale_price = train_data['SalePrice']
train_ids = train_data['Id']
test_ids = test_data['Id']
train_data.drop('SalePrice',axis=1,inplace=True)

data = pd.concat([train_data,test_data])
for feature, null_count in sorted(list(zip(data.keys(), data.isnull().sum())), key=lambda t: t[1], reverse=True):
    if null_count == 0:
        break
    print(feature, null_count)
data.fillna({'GarageType':'NA','GarageFinish':'NA','GarageQual':'NA','GarageCond':'NA'}, inplace=True)

data.fillna({'FireplaceQu':'NA'}, inplace=True)

data.fillna({'BsmtQual':'NA','BsmtCond':'NA','BsmtExposure':'NA','BsmtFinType1':'NA','BsmtFinType2':'NA'},inplace=True)

data.fillna({'PoolQC':'NA'}, inplace=True)

data.fillna({'Fence':'NA'}, inplace=True)

data.fillna({'Alley':'NA'},inplace=True)

data.fillna({'MiscFeature':'NA'}, inplace=True)

data.fillna({'MasVnrType':'None','MasVnrArea':0}, inplace=True)

data.fillna({'BsmtFullBath':0,'BsmtHalfBath':0}, inplace=True)
data.fillna({'TotalBsmtSF':0,'BsmtFinSF1':0,'BsmtFinSF2':0,'BsmtUnfSF':0},inplace=True)

data.fillna({'Utilities':'AllPub'},inplace=True)

data.fillna({'Exterior1st':'HdBoard','Exterior2nd':'Ws Sdng'}, inplace=True)

data.fillna({'Electrical':'SBrkr'},inplace=True)

data.fillna({'KitchenQual':'TA'},inplace=True)

data.fillna({'GarageCars':0,'GarageArea':0}, inplace=True)

data.fillna({'GarageYrBlt':'NA'},inplace=True)

data.fillna({'Functional':'Typ'},inplace=True)

data.fillna({'SaleType':'WD'},inplace=True)

data.fillna({'MSZoning':'RL'},inplace=True)

data.fillna({'LotFrontage':data['LotFrontage'].mean()},inplace=True)
data['BsmtFinSF1'] = np.where(data['TotalBsmtSF'] == 0.0, 0.0, data['BsmtFinSF1'] / data['TotalBsmtSF'])
data['BsmtFinSF2'] = np.where(data['TotalBsmtSF'] == 0.0, 0.0, data['BsmtFinSF2'] / data['TotalBsmtSF'])
data['BsmtUnfSF'] = np.where(data['TotalBsmtSF'] == 0.0, 0.0, data['BsmtUnfSF'] / data['TotalBsmtSF'])
# convert quality data to ordered integer
str_to_num = dict()
str_to_num['ExterQual'] = {'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
str_to_num['ExterCond'] = str_to_num['ExterQual']
str_to_num['BsmtQual'] = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
str_to_num['BsmtCond'] = str_to_num['BsmtQual']
str_to_num['BsmtExposure'] = {'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}
str_to_num['BsmtFinType1'] = {'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}
str_to_num['BsmtFinType2'] = str_to_num['BsmtFinType1']
str_to_num['HeatingQC'] = str_to_num['ExterQual']
str_to_num['KitchenQual'] = str_to_num['ExterQual']
str_to_num['FireplaceQu'] = str_to_num['BsmtQual']
str_to_num['GarageFinish'] = {'NA':0, 'Unf':1, 'RFn':2, 'Fin':3}
str_to_num['GarageQual'] = str_to_num['BsmtQual']
str_to_num['GarageCond'] = str_to_num['BsmtCond']
str_to_num['PoolQC'] = {'NA':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
str_to_num['Fence'] = {'NA':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4}

data = data.replace(str_to_num)
data['BsmtQualCond'] = data['BsmtQual'] * data['BsmtCond']

data['GarageQualCond'] = data['GarageQual'] * data['GarageCond']

data['ExteriorQualCond'] = data['ExterQual'] * data['ExterCond']

data['OverallQualCond'] = data['OverallQual'] * data['OverallCond']
data = data.astype({'MSSubClass':'category'})

#data = data.astype({'OverallQual':'category','OverallCond':'category'})

data = data.astype({'MoSold':'category','YrSold':'category','YearBuilt':'category','YearRemodAdd':'category','GarageYrBlt':'category'})
data = pd.get_dummies(data)
data.keys()
data.info()
train_X = np.array(data[data['Id'].isin(train_ids)].drop('Id',axis=1))

test_X = np.array(data[data['Id'].isin(test_ids)].drop('Id',axis=1))

train_y = np.array(sale_price)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, random_state=99, train_size=0.9)
X_test = test_X
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
dummy = DummyRegressor()
dummy.fit(X_train, np.log(y_train))
y_train_pred = dummy.predict(X_train)
y_val_pred = dummy.predict(X_val)
y_test_pred = dummy.predict(X_test)
sqrt(mean_squared_error(y_val_pred, np.log(y_val)))
from sklearn.linear_model import LassoCV
lasso = LassoCV(eps=1e-6,max_iter=100000,n_jobs=-1)
lasso.fit(X_train, np.log(y_train))
lasso.alpha_
(lasso.coef_ != 0).sum()
lasso.score(X_val, np.log(y_val))
y_val_pred = lasso.predict(X_val)

y_train_pred = lasso.predict(X_train)

y_test_pred = lasso.predict(X_test)
sqrt(mean_squared_error(y_val_pred, np.log(y_val)))
print(np.std(np.log(y_train)), np.std(y_train_pred))
print(np.std(np.log(y_val)), np.std(y_val_pred))
plt.scatter(y_train_pred, np.log(y_train) - y_train_pred, c='b')
plt.scatter(y_val_pred, np.log(y_val) - y_val_pred, c='r')
from sklearn.linear_model import RANSACRegressor, Lasso
lasso = Lasso(alpha=lasso.alpha_,max_iter=10000)
ransac = RANSACRegressor(base_estimator=lasso, min_samples=0.75, stop_score=0.98, max_trials=1000, loss='squared_loss')
ransac.fit(X_train, np.log(y_train))
ransac.n_trials_
ransac.score(X_val, np.log(y_val))
y_val_pred = ransac.predict(X_val)

y_train_pred = ransac.predict(X_train)

y_test_pred = ransac.predict(X_test)
sqrt(mean_squared_error(y_val_pred, np.log(y_val)))
print(np.std(np.log(y_train)), np.std(y_train_pred))
print(np.std(np.log(y_val)), np.std(y_val_pred))
plt.scatter(y_train_pred, np.log(y_train) - y_train_pred, c='b')
plt.scatter(y_val_pred, np.log(y_val) - y_val_pred, c='r')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_features='sqrt', criterion='mae',n_estimators=1000,n_jobs=-1)
rf.fit(X_train, np.log(y_train) - y_train_pred)
rf.score(X_val, np.log(y_val) - y_val_pred)
sqrt(mean_squared_error(rf.predict(X_val) + y_val_pred, np.log(y_val)))
sorted(list(zip(data.keys()[1:], rf.feature_importances_)), key=lambda x: x[1], reverse=True)
y_val_pred += rf.predict(X_val)

y_train_pred += rf.predict(X_train)

y_test_pred += rf.predict(X_test)
print(np.std(np.log(y_train)), np.std(y_train_pred))
print(np.std(np.log(y_val)), np.std(y_val_pred))
plt.scatter(y_train_pred, np.log(y_train) - y_train_pred, c='b')
plt.scatter(y_val_pred, np.log(y_val) - y_val_pred, c='r')
y_pred = pd.DataFrame(np.exp(y_test_pred), columns=['SalePrice'])
y_pred['Id']=test_ids
y_pred.to_csv('prediction.csv',columns=['Id','SalePrice'],index=False)
