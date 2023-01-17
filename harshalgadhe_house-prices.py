# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
f=open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt','r')

print(f.read())
df_train.replace(np.nan,'hi',inplace=True)

df_train.head()
df_test.replace(np.nan,'hi',inplace=True)

df_test.head()
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()
df_train['MSZoning']=LE.fit(df_train['MSZoning']).transform(df_train['MSZoning'])

df_train['Street']=LE.fit(df_train['Street']).transform(df_train['Street'])

df_train['Alley']=LE.fit(df_train['Alley']).transform(df_train['Alley'])

df_train['LotShape']=LE.fit(df_train['LotShape']).transform(df_train['LotShape'])

df_train['LandContour']=LE.fit(df_train['LandContour']).transform(df_train['LandContour'])

df_train['Utilities']=LE.fit(df_train['Utilities']).transform(df_train['Utilities'])

df_train['LotConfig']=LE.fit(df_train['LotConfig']).transform(df_train['LotConfig'])

df_train['LandSlope']=LE.fit(df_train['LandSlope']).transform(df_train['LandSlope'])

df_train['Neighborhood']=LE.fit(df_train['Neighborhood']).transform(df_train['Neighborhood'])

df_train['Condition1']=LE.fit(df_train['Condition1']).transform(df_train['Condition1'])

df_train['Condition2']=LE.fit(df_train['Condition2']).transform(df_train['Condition2'])

df_train['BldgType']=LE.fit(df_train['BldgType']).transform(df_train['BldgType'])

df_train['HouseStyle']=LE.fit(df_train['HouseStyle']).transform(df_train['HouseStyle'])

df_train['BldgType']=LE.fit(df_train['BldgType']).transform(df_train['BldgType'])

df_train['RoofStyle']=LE.fit(df_train['RoofStyle']).transform(df_train['RoofStyle'])

df_train['RoofMatl']=LE.fit(df_train['RoofMatl']).transform(df_train['RoofMatl'])

df_train['Exterior1st']=LE.fit(df_train['Exterior1st']).transform(df_train['Exterior1st'])

df_train['Exterior2nd']=LE.fit(df_train['Exterior2nd']).transform(df_train['Exterior2nd'])

df_train['MasVnrType']=LE.fit(df_train['MasVnrType']).transform(df_train['MasVnrType'])

df_train['Exterior2nd']=LE.fit(df_train['Exterior2nd']).transform(df_train['Exterior2nd'])

df_train['ExterQual']=LE.fit(df_train['ExterQual']).transform(df_train['ExterQual'])

df_train['ExterCond']=LE.fit(df_train['ExterCond']).transform(df_train['ExterCond'])

df_train['Foundation']=LE.fit(df_train['Foundation']).transform(df_train['Foundation'])

df_train['BsmtQual']=LE.fit(df_train['BsmtQual']).transform(df_train['BsmtQual'])

df_train['BsmtCond']=LE.fit(df_train['BsmtCond']).transform(df_train['BsmtCond'])

df_train['BsmtExposure']=LE.fit(df_train['BsmtExposure']).transform(df_train['BsmtExposure'])

df_train['BsmtFinType1']=LE.fit(df_train['BsmtFinType1']).transform(df_train['BsmtFinType1'])

df_train['BsmtFinType2']=LE.fit(df_train['BsmtFinType2']).transform(df_train['BsmtFinType2'])

df_train['Heating']=LE.fit(df_train['Heating']).transform(df_train['Heating'])

df_train['HeatingQC']=LE.fit(df_train['HeatingQC']).transform(df_train['HeatingQC'])

df_train['CentralAir']=LE.fit(df_train['CentralAir']).transform(df_train['CentralAir'])

df_train['Electrical']=LE.fit(df_train['Electrical']).transform(df_train['Electrical'])

df_train['KitchenQual']=LE.fit(df_train['KitchenQual']).transform(df_train['KitchenQual'])

df_train['Functional']=LE.fit(df_train['Functional']).transform(df_train['Functional'])

df_train['FireplaceQu']=LE.fit(df_train['FireplaceQu']).transform(df_train['FireplaceQu'])

df_train['GarageType']=LE.fit(df_train['GarageType']).transform(df_train['GarageType'])

df_train['GarageFinish']=LE.fit(df_train['GarageFinish']).transform(df_train['GarageFinish'])

df_train['GarageQual']=LE.fit(df_train['GarageQual']).transform(df_train['GarageQual'])

df_train['GarageCond']=LE.fit(df_train['GarageCond']).transform(df_train['GarageCond'])

df_train['PavedDrive']=LE.fit(df_train['PavedDrive']).transform(df_train['PavedDrive'])

df_train['PoolQC']=LE.fit(df_train['PoolQC']).transform(df_train['PoolQC'])

df_train['Fence']=LE.fit(df_train['Fence']).transform(df_train['Fence'])

df_train['MiscFeature']=LE.fit(df_train['MiscFeature']).transform(df_train['MiscFeature'])

df_train['SaleType']=LE.fit(df_train['SaleType']).transform(df_train['SaleType'])

df_train['SaleCondition']=LE.fit(df_train['SaleCondition']).transform(df_train['SaleCondition'])

df_train.head()
df_test['MSZoning']=LE.fit(df_test['MSZoning']).transform(df_test['MSZoning'])

df_test['Street']=LE.fit(df_test['Street']).transform(df_test['Street'])

df_test['Alley']=LE.fit(df_test['Alley']).transform(df_test['Alley'])

df_test['LotShape']=LE.fit(df_test['LotShape']).transform(df_test['LotShape'])

df_test['LandContour']=LE.fit(df_test['LandContour']).transform(df_test['LandContour'])

df_test['Utilities']=LE.fit(df_test['Utilities']).transform(df_test['Utilities'])

df_test['LotConfig']=LE.fit(df_test['LotConfig']).transform(df_test['LotConfig'])

df_test['LandSlope']=LE.fit(df_test['LandSlope']).transform(df_test['LandSlope'])

df_test['Neighborhood']=LE.fit(df_test['Neighborhood']).transform(df_test['Neighborhood'])

df_test['Condition1']=LE.fit(df_test['Condition1']).transform(df_test['Condition1'])

df_test['Condition2']=LE.fit(df_test['Condition2']).transform(df_test['Condition2'])

df_test['BldgType']=LE.fit(df_test['BldgType']).transform(df_test['BldgType'])

df_test['HouseStyle']=LE.fit(df_test['HouseStyle']).transform(df_test['HouseStyle'])

df_test['BldgType']=LE.fit(df_test['BldgType']).transform(df_test['BldgType'])

df_test['RoofStyle']=LE.fit(df_test['RoofStyle']).transform(df_test['RoofStyle'])

df_test['RoofMatl']=LE.fit(df_test['RoofMatl']).transform(df_test['RoofMatl'])

df_test['Exterior1st']=LE.fit(df_test['Exterior1st']).transform(df_test['Exterior1st'])

df_test['Exterior2nd']=LE.fit(df_test['Exterior2nd']).transform(df_test['Exterior2nd'])

df_test['MasVnrType']=LE.fit(df_test['MasVnrType']).transform(df_test['MasVnrType'])

df_test['Exterior2nd']=LE.fit(df_test['Exterior2nd']).transform(df_test['Exterior2nd'])

df_test['ExterQual']=LE.fit(df_test['ExterQual']).transform(df_test['ExterQual'])

df_test['ExterCond']=LE.fit(df_test['ExterCond']).transform(df_test['ExterCond'])

df_test['Foundation']=LE.fit(df_test['Foundation']).transform(df_test['Foundation'])

df_test['BsmtQual']=LE.fit(df_test['BsmtQual']).transform(df_test['BsmtQual'])

df_test['BsmtCond']=LE.fit(df_test['BsmtCond']).transform(df_test['BsmtCond'])

df_test['BsmtExposure']=LE.fit(df_test['BsmtExposure']).transform(df_test['BsmtExposure'])

df_test['BsmtFinType1']=LE.fit(df_test['BsmtFinType1']).transform(df_test['BsmtFinType1'])

df_test['BsmtFinType2']=LE.fit(df_test['BsmtFinType2']).transform(df_test['BsmtFinType2'])

df_test['Heating']=LE.fit(df_test['Heating']).transform(df_test['Heating'])

df_test['HeatingQC']=LE.fit(df_test['HeatingQC']).transform(df_test['HeatingQC'])

df_test['CentralAir']=LE.fit(df_test['CentralAir']).transform(df_test['CentralAir'])

df_test['Electrical']=LE.fit(df_test['Electrical']).transform(df_test['Electrical'])

df_test['KitchenQual']=LE.fit(df_test['KitchenQual']).transform(df_test['KitchenQual'])

df_test['Functional']=LE.fit(df_test['Functional']).transform(df_test['Functional'])

df_test['FireplaceQu']=LE.fit(df_test['FireplaceQu']).transform(df_test['FireplaceQu'])

df_test['GarageType']=LE.fit(df_test['GarageType']).transform(df_test['GarageType'])

df_test['GarageFinish']=LE.fit(df_test['GarageFinish']).transform(df_test['GarageFinish'])

df_test['GarageQual']=LE.fit(df_test['GarageQual']).transform(df_test['GarageQual'])

df_test['GarageCond']=LE.fit(df_test['GarageCond']).transform(df_test['GarageCond'])

df_test['PavedDrive']=LE.fit(df_test['PavedDrive']).transform(df_test['PavedDrive'])

df_test['PoolQC']=LE.fit(df_test['PoolQC']).transform(df_test['PoolQC'])

df_test['Fence']=LE.fit(df_test['Fence']).transform(df_test['Fence'])

df_test['MiscFeature']=LE.fit(df_test['MiscFeature']).transform(df_test['MiscFeature'])

df_test['SaleType']=LE.fit(df_test['SaleType']).transform(df_test['SaleType'])

df_test['SaleCondition']=LE.fit(df_test['SaleCondition']).transform(df_test['SaleCondition'])

df_test.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(18,16))

df_train.corr()
df_train.columns
from sklearn.linear_model import LinearRegression

LR=LinearRegression()
df_train.replace('hi',np.nan,inplace=True)

df_train.dropna(inplace=True)
Z=df_train.drop(['Id','SalePrice'],axis=1)

y=df_train['SalePrice']

LR.fit(Z,y)
from sklearn.metrics import r2_score

yhat_train=LR.predict(Z)

print("r2_score is :",r2_score(yhat_train,y))
df_test.replace('hi',np.nan,inplace=True)

df_final=pd.DataFrame()

df_final['Id']=df_test['Id']
df_test=df_test.fillna(df_test.mean())

Z=df_test.drop(['Id'],axis=1)

yhat=LR.predict(Z)

df_final['SalePrice']=yhat

df_final
df_final.to_csv(r'C:\Users\Harshal\Desktop\HousePrices.csv', index = False)
df_final.info()