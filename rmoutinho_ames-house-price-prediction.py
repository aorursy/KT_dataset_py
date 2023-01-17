# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import statsmodels.regression.linear_model as sm

from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer



DATA_PATH = '/kaggle/input/house-prices-advanced-regression-techniques/'

train = pd.read_csv(DATA_PATH+'train.csv')

test = pd.read_csv(DATA_PATH+'test.csv')
fullset = pd.concat([train,test],axis=0,sort=False,ignore_index=True)



###

###  PREPROCESSING STARTS HERE

###



#

#  Identify features with nulls

def getFeatsWithNulls():

    featsWithNulls = [] 

    for feat in fullset.columns:

        if (fullset[feat].isnull().sum() > 0):

            featsWithNulls = featsWithNulls + [feat]

    return featsWithNulls



#

# Too much nulls on these. Let´s just drop'em

featsToDrop = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']

fullset.drop(columns=featsToDrop, inplace=True)



#

# Let's handle the garages first

for carFeat in ['GarageType','GarageFinish','GarageQual','GarageCond']:

    fullset[carFeat].fillna(value='NoG', inplace=True)

fullset.GarageYrBlt.fillna(fullset.GarageYrBlt.median(), inplace=True)

CarsInDetchdGarage = fullset[fullset.GarageType == 'Detchd']['GarageCars'].median()

fullset.GarageCars.fillna(value=CarsInDetchdGarage, inplace=True)

fullset.GarageArea.fillna(value=fullset.GarageArea.median())



#

# Now let's handle the basement

for bsmtFeat in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']:

    fullset[bsmtFeat].fillna('NoB', inplace=True)



#

# Finally, the rest

NumericFeatsWithNull = [ feat for feat in fullset.select_dtypes(exclude='object') \

                        if (fullset[feat].isnull().sum() > 0) & (feat != 'SalePrice') ]

for feat in NumericFeatsWithNull:

    fullset[feat].fillna(fullset[feat].median(), inplace=True)

fullset.MasVnrType.fillna('None', inplace=True)



FeatsWithNull = getFeatsWithNulls()

for feat in FeatsWithNull:

    if (feat != 'SalePrice'):

        most_common = fullset[feat].value_counts().index[0]

        fullset[feat].fillna(value=most_common, inplace=True)



#

# Now for some categorical feature encoding

CatFeatsToEncode = ['MSZoning','LotShape','LandContour','LotConfig','Neighborhood',\

                    'Condition1','LandSlope','HouseStyle']

for feat in CatFeatsToEncode:

    ct = ColumnTransformer([(feat, OneHotEncoder(sparse=False), ['Neighborhood'])],

                       remainder='drop')

    encFeat = pd.DataFrame(data=ct.fit_transform(fullset), columns=ct.get_feature_names())

    fullset = pd.concat([encFeat,fullset],axis=1)

fullset.drop(columns=CatFeatsToEncode, inplace=True)



#

# Drop catgorical features we won't use on the model

CatFeatsToDrop = fullset.select_dtypes('object').columns

fullset.drop(columns=CatFeatsToDrop, inplace=True)



train = fullset[fullset.SalePrice.isnull() != True].copy()

test = fullset[fullset.SalePrice.isnull() == True].copy()

test.drop(columns='SalePrice', inplace=True)

###

###  PREPROCESSING ENDS HERE

###    



regressor = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)

regressor.fit(train.drop(columns=['Id','SalePrice']), train['SalePrice'])



PricePred = regressor.predict(test.drop(columns='Id'))

res = pd.DataFrame(data = test.Id)

res['SalePrice'] = PricePred



res.to_csv('submission.csv',index=False)