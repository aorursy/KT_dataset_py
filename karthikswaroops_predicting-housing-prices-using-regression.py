# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/test.csv')
df2=pd.read_csv('../input/train.csv')

df2.head()
df2=df2.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

df2=df2.dropna()

df2.head()



#Dropping the id

df2=df2.drop('Id',axis=1)

df2.head()



df2['MSZoning']=pd.get_dummies(df2['MSZoning'])
df2['Street']=pd.get_dummies(df2['Street'])
df2['LotShape']=pd.get_dummies(df2['LotShape'])
df2['LandContour']=pd.get_dummies(df2['LandContour'])
df2['Utilities']=pd.get_dummies(df2['Utilities'])
df2['LotConfig']=pd.get_dummies(df2['LotConfig'])
df2['LandSlope']=pd.get_dummies(df2['LandSlope'])
df2['Neighborhood']=pd.get_dummies(df2['Neighborhood'])
df2['Condition1']=pd.get_dummies(df2['Condition1'])
df2['Condition2']=pd.get_dummies(df2['Condition2'])
df2['BldgType']=pd.get_dummies(df2['BldgType'])
df2['HouseStyle']=pd.get_dummies(df2['HouseStyle'])
df2['RoofStyle']=pd.get_dummies(df2['RoofStyle'])
df2['RoofMatl']=pd.get_dummies(df2['RoofMatl'])
df2['Exterior1st']=pd.get_dummies(df2['Exterior1st'])
df2['Exterior2nd']=pd.get_dummies(df2['Exterior2nd'])
df2['MasVnrType']=pd.get_dummies(df2['MasVnrType'])
df2['ExterQual']=pd.get_dummies(df2['ExterQual'])

df2['ExterCond']=pd.get_dummies(df2['ExterCond'])
df2['Foundation']=pd.get_dummies(df2['Foundation'])
df2['BsmtQual']=pd.get_dummies(df2['BsmtQual'])
df2['BsmtCond']=pd.get_dummies(df2['BsmtCond'])
df2['BsmtExposure']=pd.get_dummies(df2['BsmtExposure'])
df2['BsmtFinType1']=pd.get_dummies(df2['BsmtFinType1'])
df2['BsmtFinType2']=pd.get_dummies(df2['BsmtFinType2'])
df2['Heating']=pd.get_dummies(df2['Heating'])
df2['HeatingQC']=pd.get_dummies(df2['HeatingQC'])
df2['CentralAir']=pd.get_dummies(df2['CentralAir'])
df2['Electrical']=pd.get_dummies(df2['Electrical'])
df2['KitchenQual']=pd.get_dummies(df2['KitchenQual'])
df2['Functional']=pd.get_dummies(df2['Functional'])
df2['FireplaceQu']=pd.get_dummies(df2['FireplaceQu'])
df2['GarageType']=pd.get_dummies(df2['GarageType'])
df2['GarageFinish']=pd.get_dummies(df2['GarageFinish'])
df2['GarageQual']=pd.get_dummies(df2['GarageQual'])
df2['GarageCond']=pd.get_dummies(df2['GarageCond'])
df2['PavedDrive']=pd.get_dummies(df2['PavedDrive'])
df2['SaleType']=pd.get_dummies(df2['SaleType'])
df2['SaleCondition']=pd.get_dummies(df2['SaleCondition'])
df2.head()
x=df2.drop('SalePrice',axis=1)

y=df2['SalePrice']
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
lr.fit(x_train,y_train)

predictions=lr.predict(x_test)
predictions