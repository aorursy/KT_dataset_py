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
import numpy as np

import pandas as pd

data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

missing=data_train.isnull().sum()

missing_column=[]

for i in range(0,len(missing)):

    if missing[i]!=0:

       missing_column.append(missing.index[i])
for i in missing_column:

    if i=="Electrical":

        data_train[i].fillna("SBrkr", inplace=True) 

    elif i=='LotFrontage':

        data_train[i].fillna(0, inplace=True)

    elif i == 'MasVnrArea':

        data_train[i].fillna(0, inplace=True)

    elif i =='GarageYrBlt':

        data_train[i].fillna(0, inplace=True)

    else :

        data_train[i].fillna("0", inplace=True)
data_train=data_train.iloc[:,1:] #droping "id"

data_train_X=data_train.iloc[:,:-1]

data_train_Y=data_train.iloc[:,-1]
data_train_X.columns.to_series().groupby(data_train.dtypes).groups
not_catgorical=['MSSubClass','LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',

                '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','MasVnrArea', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                'EnclosedPorch','OverallQual', 'OverallCond','GarageYrBlt', '3SsnPorch','YrSold','YearBuilt', 'YearRemodAdd', 'PoolArea']

len (not_catgorical)

catgorical=[ 'MSZoning', 'Street', 'Alley',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC',

       'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond',

       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',

       'MoSold',  'SaleType', 'SaleCondition']
onehotencoder=pd.get_dummies(data_train[catgorical],drop_first=True)
data_train_X=pd.concat([onehotencoder, data_train[not_catgorical]], axis=1,ignore_index=True)
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train, y_test = train_test_split(data_train_X, data_train_Y, test_size=0.2,random_state=0)

import statsmodels.api as sm
X_train=np.append(arr=np.ones((1168,1)).astype(int),values=X_train,axis=1 ) 
regressor_OLS=sm.OLS(y_train,X_train, missing='none').fit()

regressor_OLS.summary()
def backwardElimination(x, SL):

    numVars = len(x[0])

    temp = np.zeros((1168,260)).astype(int)

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(y_train,x).fit()

        maxVar = max(regressor_OLS.pvalues)

        adjR_before = regressor_OLS.rsquared_adj.astype(float)

        if maxVar > SL:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    temp[:,j] = x[:, j]

                    x = np.delete(x, j, 1)

                    tmp_regressor = sm.OLS(y_train, x).fit()

                    adjR_after = tmp_regressor.rsquared_adj.astype(float)

                    if (adjR_before >= adjR_after):

                        x_rollback = np.hstack((x, temp[:,[0,j]]))

                        x_rollback = np.delete(x_rollback, j, 1)

                        print (regressor_OLS.summary())

                        return x_rollback

                    else:

                        continue

    regressor_OLS.summary()

    return x

 

SL = 0.5

X_Modeled = backwardElimination(X_train, SL)
print (X_Modeled.shape, X_train.shape)
"""As we see, shape of data are identical, as well as  Adj. R-squared, 

so automatic backward Elimination code I found in internet seems not working :/  

But result: Adj. R-squared:0.930, is not bad. Am I wrong? 

How to automate backward Elimination? """