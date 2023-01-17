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
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df.info()
df=df.drop(['MiscFeature','PoolQC','Alley'],axis=1)
df.info()
df['Utilities'].nunique()
x=pd.get_dummies(df['Utilities'])
x=x.drop('AllPub',axis=1)
df['Ut']=x
df=df.drop('Utilities',axis=1)

df['Fence'].value_counts()
df['F']=df['Fence'].fillna(0)
t=pd.get_dummies(df['Fence'])
t
df=pd.concat([df,t],axis=1)
df.info()
df=df.drop(['Fence','F'],axis=1)
df.info()
df.describe()
df['GarageQual']=df['GarageQual'].fillna(0)
k=pd.get_dummies(df['GarageQual'])
df['GarageCond']=df['GarageCond'].fillna(0)
l=pd.get_dummies(df['GarageCond'])
df['GarageType']=df['GarageType'].fillna(0)
m=pd.get_dummies(df['GarageType'])
df['GarageFinish'].value_counts()
df['GarageFinish']=df['GarageFinish'].fillna(0)
n=pd.get_dummies(df['GarageFinish'])
df['GarageYrBlt']=df['GarageYrBlt'].fillna(2005.0)

df['LotFrontage']=df['LotFrontage'].fillna(70)
df.info()
df['MasVnrArea'].fillna(0)
df['MasVnrType'].value_counts()
df['MasVnrType']=df['MasVnrType'].fillna(0)
p=pd.get_dummies(df['MasVnrType'])
df=pd.concat([df,k,l,m,n,p],axis=1)
df=df.drop(['MasVnrType','GarageFinish','GarageType','GarageCond','GarageQual'],axis=1)
df['RoofMatl'].value_counts()
df['RoofMatl']=df['RoofMatl'].fillna(0)
x=pd.get_dummies(df['RoofMatl'])
df=pd.concat([df,x],axis=1)
df=df.drop('RoofMatl',axis=1)

