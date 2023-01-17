# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

from numpy import *

from scipy import stats

from scipy.interpolate import *

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.drop(['PoolQC','Alley','Fence','MiscFeature','GarageYrBlt'],axis=1,inplace=True)
df.fillna(df.mean(),inplace=True)

df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

df.dropna(inplace=True)
df.drop(['Id'],axis=1,inplace=True)
columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

        'GarageCond', 'PavedDrive','SaleCondition','SaleType']
dft = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
dft.drop(['PoolQC','Fence','Alley','MiscFeature','GarageYrBlt'],axis=1,inplace=True)
dft.drop(['Id'],axis=1,inplace=True)
dft.fillna(dft.mean(),inplace=True)

dft['FireplaceQu'] = dft['FireplaceQu'].fillna(dft['FireplaceQu'].mode()[0])

dft['GarageType']=dft['GarageType'].fillna(dft['GarageType'].mode()[0])

dft['MasVnrType']=dft['MasVnrType'].fillna(dft['MasVnrType'].mode()[0])

dft['MasVnrArea']=dft['MasVnrArea'].fillna(dft['MasVnrArea'].mode()[0])

dft['BsmtQual']=dft['BsmtQual'].fillna(dft['BsmtQual'].mode()[0])

dft['BsmtCond']=dft['BsmtCond'].fillna(dft['BsmtCond'].mode()[0])

dft['BsmtExposure']=dft['BsmtExposure'].fillna(dft['BsmtExposure'].mode()[0])

dft['BsmtFinType1']=dft['BsmtFinType1'].fillna(dft['BsmtFinType1'].mode()[0])

dft['BsmtFinType2']=dft['BsmtFinType2'].fillna(dft['BsmtFinType2'].mode()[0])

dft['GarageFinish']=dft['GarageFinish'].fillna(dft['GarageFinish'].mode()[0])

dft['GarageQual']=dft['GarageQual'].fillna(dft['GarageQual'].mode()[0])

dft['GarageCond']=dft['GarageCond'].fillna(dft['GarageCond'].mode()[0])

dft['MSZoning']=dft['MSZoning'].fillna(dft['MSZoning'].mode()[0])

dft['Utilities']=dft['Utilities'].fillna(dft['Utilities'].mode()[0])

dft['Exterior1st']=dft['Exterior1st'].fillna(dft['Exterior1st'].mode()[0])

dft['Exterior2nd']=dft['Exterior2nd'].fillna(dft['Exterior2nd'].mode()[0])

dft['KitchenQual']=dft['KitchenQual'].fillna(dft['KitchenQual'].mode()[0])

dft['Functional']=dft['Functional'].fillna(dft['Functional'].mode()[0])

dft['SaleType']=dft['SaleType'].fillna(dft['SaleType'].mode()[0])

dft['BsmtFullBath']=dft['BsmtFullBath'].fillna(dft['BsmtFullBath'].mode()[0])

dft['BsmtHalfBath']=dft['BsmtHalfBath'].fillna(dft['BsmtHalfBath'].mode()[0])
def onehot (multcolumns):

    dffinal = finaldf

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(finaldf[fields],drop_first=True)

        

        finaldf.drop([fields],axis=1,inplace=True)

        if i==0:

            dffinal = df1.copy()

        else:

            dffinal = pd.concat([dffinal,df1],axis=1)

        i=i+1

    dffinal = pd.concat([finaldf,dffinal],axis=1)

    return dffinal
finaldf =pd.concat([df,dft],axis=0)
finaldf = onehot(columns)
finaldf.shape
finaldf = finaldf.loc[:,~finaldf.columns.duplicated()]
#df.columns.to_series().groupby(df.dtypes).groups
dftrain = finaldf.iloc[:1422,:]

dftest = finaldf.iloc[1422:,:]
dftest.drop(['SalePrice'],axis=1,inplace=True)
mse = mean_squared_error
Xtrain = dftrain.drop(['SalePrice'],axis=1)

Ytrain = dftrain['SalePrice']
booster = ['gbtree','gblinear']

base_score = [0.25,0.5,0.75,1]

n_estimators = [100,500,900,1100,1300,1500]

max_depth = [2,3,5,9,11,15]

learning_rate =[0.05,1,0.15,0.20]

min_child_weight = [1,2,3,4]



hyperparameter_grid = {

    'n_estimators':n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score,

}
xg = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.05, max_delta_step=0,

             max_depth=2, min_child_weight=4, missing=None, n_estimators=1100,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xg.fit(Xtrain,Ytrain)
random_cv = RandomizedSearchCV(estimator=xg,

                              param_distributions=hyperparameter_grid,

                              cv=5,n_iter=50,

                              scoring='neg_mean_absolute_error',n_jobs=4,

                              verbose=5,

                              return_train_score=True,

                              random_state=1)
yp = xg.predict(Xtrain)
xg.score(Xtrain,Ytrain)
mse(Ytrain,yp)
yresult = xg.predict(dftest)
pred = pd.DataFrame(yresult)

sub_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

dataset = pd.concat([sub_df['Id'],pred],axis=1)

dataset.columns=['Id','SalePrice']

dataset.to_csv('sample_submission.csv',index=False)