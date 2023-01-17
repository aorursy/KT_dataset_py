# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



frame = (train, test)

df1 = pd.concat(frame, keys = ['train','test'])
df1.corr().sum().sort_values(ascending=False)
df2 = df1.sort_values(by = 'GrLivArea')

df3 = df2.isnull().sum().sort_values(ascending= False)



#Find values with NaN/missing values

df3 = df2.isnull().sum().sort_values(ascending= False)

df3.head(40)



# First lets fill columns that have less than 50% NaN/missing values 



for column in ['LotFrontage', 'GarageQual', 'GarageYrBlt', 'GarageFinish',

       'GarageCond', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual',

       'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'MasVnrArea', 'MSZoning',

       'Utilities', 'Functional', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea',

       'BsmtFinSF2', 'Exterior1st', 'TotalBsmtSF', 'GarageCars', 'BsmtUnfSF',

       'Electrical','BsmtFinSF1','KitchenQual','SaleType','Exterior2nd']:

    df2[column].fillna(method='ffill',inplace=True)
# Select the top 6 from the list above and drop em since they have over 50% NaN/missing values



df3.index[0:6]

df2.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'],axis=1, inplace=True)
df2.columns
# Choose as many numerical features



fig = plt.figure(figsize=(12,5)) 



fig.add_subplot(221)            

plt.scatter(df2.TotalBsmtSF, y=df2.SalePrice)



fig.add_subplot(222)         

plt.scatter(df2.LotArea, y=df2.SalePrice)               



fig.add_subplot(223)            

plt.scatter(df2.LotFrontage, y=df2.SalePrice)



fig.add_subplot(224)            

plt.scatter(df2.MasVnrArea, y=df2.SalePrice)



plt.show()                      
# Drop outliers



columns1 = ['LotFrontage','LotArea','TotalBsmtSF','MasVnrArea']

df2[columns1].sort_values(by = columns1,ascending=False)[0:4]



df2.drop(df2[df2['Id'] == 1298].index, inplace = True)

df2.drop(df2[df2['Id'] == 386].index, inplace = True)

df2.drop(df2[df2['Id'] == 934].index, inplace = True)

df2.drop(df2[df2['Id'] == 313].index, inplace = True)

# Convert Categorical features to dummy/numerical variables

df2_dummy = pd.get_dummies(df2)
Id =  df2_dummy.loc['test']

dftest = df2_dummy.loc['test']

dftrain = df2_dummy.loc['train']

Id = Id.Id
# Drop any NaN values

dftrain.dropna(axis = 0, inplace = True)

dftest.drop(['SalePrice','Id'], axis =1, inplace=True)
from sklearn.model_selection import train_test_split

import xgboost 



dftrain_X = dftrain.drop(['SalePrice','Id'], axis = 1)

dftrain_y = dftrain['SalePrice']
# I used hypertuning to get the optimal parameters. Left it out for simplicity.



xgboost = xgboost.XGBRegressor(learning_rate=0.05,  

                      colsample_bytree = 0.5,

                      subsample = 0.8,

                      n_estimators=1000, 

                      max_depth=5, 

                      gamma=5)



xgboost.fit(dftrain_X, dftrain_y)

# Save it

predictions= xgboost.predict(dftest)

solution = pd.DataFrame({"Id": Id, 'SalePrice': predictions})

solution.to_csv('house_pred.csv', index=False)