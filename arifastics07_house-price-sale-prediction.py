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
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
print(train.info())

print(test.info())
train.head()
frames = [train, test]

full = pd.concat(frames)

full.head()
full1 = full.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','Id'])

full1.head()
full1.info()
from scipy.stats import mode



full1['BsmtCond'] = full1['BsmtCond'].fillna(full1['BsmtCond'].mode()[0])

full1['BsmtExposure'] = full1['BsmtExposure'].fillna(full1['BsmtExposure'].mode()[0])

full1['BsmtFinSF1'] = full1['BsmtFinSF1'].fillna(full1['BsmtFinSF1'].mean())

full1['BsmtFinSF2'] = full1['BsmtFinSF2'].fillna(full1['BsmtFinSF2'].mean())

full1['BsmtFinType1'] = full1['BsmtFinType1'].fillna(full1['BsmtFinType1'].mode()[0])

full1['BsmtFinType2'] = full1['BsmtFinType2'].fillna(full1['BsmtFinType2'].mode()[0])

full1['BsmtFullBath'] = full1['BsmtFullBath'].fillna(full1['BsmtFullBath'].mean())

full1['BsmtHalfBath'] = full1['BsmtHalfBath'].fillna(full1['BsmtHalfBath'].mean())

full1['BsmtQual'] = full1['BsmtQual'].fillna(full1['BsmtQual'].mode()[0])

full1['BsmtUnfSF'] = full1['BsmtUnfSF'].fillna(full1['BsmtUnfSF'].mean())

full1['Electrical'] = full1['Electrical'].fillna(full1['Electrical'].mode()[0])

full1['Exterior1st'] = full1['Exterior1st'].fillna(full1['Exterior1st'].mode()[0])

full1['Exterior2nd'] = full1['Exterior2nd'].fillna(full1['Exterior2nd'].mode()[0])

full1['Functional'] = full1['Functional'].fillna(full1['Functional'].mode()[0])

full1['GarageArea'] = full1['GarageArea'].fillna(full1['GarageArea'].mean())

full1['GarageCars'] = full1['GarageCars'].fillna(full1['GarageCars'].mean())

full1['GarageCond'] = full1['GarageCond'].fillna(full1['GarageCond'].mode()[0])

full1['GarageFinish'] = full1['GarageFinish'].fillna(full1['GarageFinish'].mode()[0])

full1['GarageQual'] = full1['GarageQual'].fillna(full1['GarageQual'].mode()[0])

full1['GarageType'] = full1['GarageType'].fillna(full1['GarageType'].mode()[0])

full1['GarageYrBlt'] = full1['GarageYrBlt'].fillna(full1['GarageYrBlt'].mean())

full1['KitchenQual'] = full1['KitchenQual'].fillna(full1['KitchenQual'].mode()[0])

full1['LotFrontage'] = full1['LotFrontage'].fillna(full1['LotFrontage'].mean())

full1['MSZoning'] = full1['MSZoning'].fillna(full1['MSZoning'].mode()[0])

full1['MasVnrArea'] = full1['MasVnrArea'].fillna(full1['MasVnrArea'].mean())

full1['MasVnrType'] = full1['MasVnrType'].fillna(full1['MasVnrType'].mode()[0])

full1['TotalBsmtSF'] = full1['TotalBsmtSF'].fillna(full1['TotalBsmtSF'].mean())

full1['SaleType'] = full1['SaleType'].fillna(full1['SaleType'].mode()[0])

full1['Utilities'] = full1['Utilities'].fillna(full1['Utilities'].mode()[0])

full2 = pd.get_dummies(full1, drop_first=True)

full2
full2.info()
print(train.shape)

print(test.shape)
df_train = full2.iloc[:1460]

df_test = full2.iloc[1460:]
print(df_train.shape)

print(df_test.shape)
X = df_train.iloc[:, df_train.columns != 'SalePrice']

y = df_train.iloc[:, df_train.columns == 'SalePrice']

y_log = np.log(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn import svm, naive_bayes, tree, ensemble, linear_model

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

import lightgbm as lgb

from math import sqrt
model = CatBoostRegressor(logging_level='Silent')

model.fit(X_train, y_train)



print(sqrt(mean_squared_error(y_test, model.predict(X_test))))

print(sqrt(mean_squared_log_error(y_test, model.predict(X_test))))
#svm.lin     58882.6002   0.3408

#svm.nu      87798.5648   0.4361

#svm.svr     88653.0630   0.4322

#nb.bern     72825.9832   0.3141

#nb.gaus     52054.6709   0.2348

#nb.mult     65160.7555   0.3091

#tr.dtr      41914.0813   0.2327

#tr.etr      52473.6505   0.2278

#eb.ada      35580.0614   0.2203

#eb.bag      30194.3135   0.1592

#eb.etr      28310.6934   0.1562

#eb.gbr      27704.2617   0.1406

#eb.rf       28578.2555   0.1530

#lm.ardr     31042.8823   0.1615

#lm.hub      42824.2604   0.2166

#lm.pass     50180.6378   0.2648

#lm.rid      29999.3980   0.1612

#xgb         28236.5141   0.1429

#cb          27133.8871   0.1367

#lgb         29143.2089   0.1460

#cb7_.1_1,5  25503.2592   0.1379
testing = df_test.iloc[:, df_test.columns != 'SalePrice']
importances = model.feature_importances_

a = (importances)

b = (X.columns)



fr = [a,b]



fi = pd.DataFrame({'var':b,

                  'imp':a})

fi
fi.sort_values(by=['imp'])
new_var = fi.loc[fi['imp'] > 0.5]

new_var
X_new = X[['1stFlrSF','2ndFlrSF','BedroomAbvGr','BsmtFinSF1','BsmtFullBath','BsmtUnfSF','Fireplaces','FullBath','GarageArea','GarageCars','GarageYrBlt','GrLivArea','HalfBath','LotArea','LotFrontage','MSSubClass','OpenPorchSF','OverallCond','OverallQual','TotRmsAbvGrd','TotalBsmtSF','WoodDeckSF','YearBuilt','YearRemodAdd','BsmtQual_Gd','BsmtQual_TA','CentralAir_Y','Condition1_Norm','ExterQual_Gd','ExterQual_TA','GarageFinish_Unf','KitchenQual_Gd','Neighborhood_Edwards']]

testing_new = testing[['1stFlrSF','2ndFlrSF','BedroomAbvGr','BsmtFinSF1','BsmtFullBath','BsmtUnfSF','Fireplaces','FullBath','GarageArea','GarageCars','GarageYrBlt','GrLivArea','HalfBath','LotArea','LotFrontage','MSSubClass','OpenPorchSF','OverallCond','OverallQual','TotRmsAbvGrd','TotalBsmtSF','WoodDeckSF','YearBuilt','YearRemodAdd','BsmtQual_Gd','BsmtQual_TA','CentralAir_Y','Condition1_Norm','ExterQual_Gd','ExterQual_TA','GarageFinish_Unf','KitchenQual_Gd','Neighborhood_Edwards']]
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
model1 = CatBoostRegressor(logging_level='Silent')

model1.fit(Xn_train, yn_train)



print(sqrt(mean_squared_error(yn_test, model1.predict(Xn_test))))

print(sqrt(mean_squared_log_error(yn_test, model1.predict(Xn_test))))
predict = model1.predict(testing_new)
id_col = sub['Id']

submission = pd.DataFrame({'Id':id_col,

                         'SalePrice':predict})

submission.head()
submission.to_csv('sub_cb_var.csv', index=False)