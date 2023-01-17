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
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler
train_path = '../input/home-data-for-ml-course/train.csv'

test_path = '../input/home-data-for-ml-course/test.csv'



raw_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)



raw_data.head()

X1 = raw_data.iloc[:,:-1]

Y = raw_data.iloc[:,-1]



X2 = test_data.iloc[:,:]



X_full = pd.concat(objs=[X1, X2], axis=0)

X_full.describe(include = 'all')



print(X_full.shape)

null_columns=X_full.columns[X_full.isnull().any()]



X_full[null_columns].isnull().sum()
X_full[null_columns].dtypes
Object_type = [

'MSZoning','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','BsmtQual',

'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical'

,'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',

'GarageQual','GarageCond','PoolQC','Fence','MiscFeature','SaleType']



X_full.fillna( {x: 'NA' for x in Object_type}, inplace = True) 



int_type = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

            'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']



X_full.fillna( {x: 0 for x in int_type}, inplace = True) 





X_full[null_columns].isnull().sum()

numeric_list = list(X_full.select_dtypes(include = [np.number]).columns.values)



scaler = StandardScaler()

X_full[numeric_list] = scaler.fit_transform(X_full[numeric_list])
X_full.head()
X_full_dummies = pd.get_dummies(X_full , drop_first=True)

X_full_dummies.head()
X_train_full = X_full_dummies.iloc[:1460, 1:]

X_test_full = X_full_dummies.iloc[1460:, 1:]



print(X_train_full.shape)

print(X_test_full.shape)
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV



lasso = Lasso()



parameters_val = {'alpha': [145, 150,155,157,250] }



lasso_regresor = GridSearchCV(lasso , parameters_val , scoring = 'neg_mean_absolute_error' , cv = 10)



lasso_regresor.fit(X_train_full,Y)



print(lasso_regresor.best_params_)

print(lasso_regresor.best_score_)
lassofit = Lasso(alpha = 155)

lassofit.fit(X_train_full,Y)



test_preds = lassofit.predict(X_test_full)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.head()
output.to_csv('submission.csv', index=False)