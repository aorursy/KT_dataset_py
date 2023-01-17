# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing interesting functions from several packages

import seaborn as sns

import pandas_profiling

from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import datasets

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import RobustScaler

from scipy import stats

from scipy.stats import zscore
# loading data in the project

test_path='/kaggle/input/home-data-for-ml-course/test.csv'

train_path='/kaggle/input/home-data-for-ml-course/train.csv'

sample_path='/kaggle/input/home-data-for-ml-course/sample_submission.csv'

test=pd.read_csv(test_path, index_col=0)

train=pd.read_csv(train_path, index_col=0)

sample = pd.read_csv(sample_path)
# dropping outliers

train=train[train['GrLivArea'] < 4500]

# concatenate training and testing data

X = pd.concat([train.drop("SalePrice", axis=1), test])

y_train = np.log(train["SalePrice"])
nans = X.isna().sum().sort_values(ascending=False)

nans = nans[nans > 0]

fig, ax = plt.subplots(figsize=(10, 6))

ax.grid()

ax.bar(nans.index, nans.values, zorder=2, color="#3f72af")

ax.set_ylabel("No. of missing values", labelpad=10)

ax.set_xlim(-0.6, len(nans) - 0.4)

ax.xaxis.set_tick_params(rotation=90)

plt.show()
cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageQual','GarageFinish',

       'GarageType','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1']

X[cols]=X[cols].fillna('None')

cols = ['GarageYrBlt','MasVnrArea','BsmtHalfBath','BsmtFullBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

       'TotalBsmtSF','GarageCars']

X[cols]=X[cols].fillna(0)

cols=['MasVnrType','MSZoning','Utilities','Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual',

     'Functional']

X[cols]=X.groupby('Neighborhood')[cols].transform(lambda x: x.fillna(x.mode()[0]))

cols=['GarageArea','LotFrontage']

X[cols]=X.groupby('Neighborhood')[cols].transform(lambda x: x.fillna(x.median()))

X['TotalSF']=X['GrLivArea']+X['TotalBsmtSF']

X['TotalPorchSF']=X['OpenPorchSF']+X['EnclosedPorch']+X['3SsnPorch']+X['ScreenPorch']

X['TotalBath']=X['FullBath']+X['BsmtFullBath']+ 0.5 * (X['BsmtHalfBath']+X['HalfBath'])

cols = ['MSSubClass','YrSold']

X[cols]=X[cols].astype('category')

cols = X.select_dtypes(np.number).columns

X[cols]=RobustScaler().fit_transform(X[cols])
X = pd.get_dummies(X)
X_train = X.loc[train.index]

X_test = X.loc[test.index]
print(X_train.info(), X_test.info())
#X['SinMoSold']=np.sin(2* np.pi * X['MoSold']/12)

#X['CosMoSold']=np.cos(2* np.pi * X['MoSold']/12)

#X=X.drop('MoSold', axis=1)
home_model = RandomForestRegressor(random_state=1)

home_model.fit(X_train,y_train)

home_preds = home_model.predict(X_train)

mae = mean_absolute_error(home_preds, y_train)

print(mae)
preds = home_model.predict(X_test)

submission = pd.DataFrame({'Id':sample['Id'],'SalePrice':np.exp(preds)})

submission.to_csv('submission.csv', index=False)
