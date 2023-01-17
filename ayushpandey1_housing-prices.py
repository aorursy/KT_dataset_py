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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_df.head()
train_id=train_df['Id']

test_id=test_df['Id']

train_df.drop('Id',axis=1,inplace=True)

test_df.drop('Id',axis=1,inplace=True)

train_df.info()
fig,ax=plt.subplots()

ax.scatter(x=train_df['GrLivArea'],y=train_df['SalePrice'])

plt.show()
# We can see the two big outliers after x coordinate=4500 lets remove

train_df.drop(train_df[(train_df['SalePrice']<200000)&(train_df['GrLivArea']>4500)].index,inplace=True)

fig,ax=plt.subplots()

ax.scatter(x=train_df['GrLivArea'],y=train_df['SalePrice'])

plt.show()
# Cobining training and testing data

all_data=pd.concat((train_df,test_df)).reset_index(drop=True)

all_data.drop('SalePrice',axis=1,inplace=True)

print(all_data.shape)
all_miss_data=((all_data.isnull().sum())/len(all_data))*100

all_miss_data=all_miss_data.drop(all_miss_data[all_miss_data==0].index).sort_values(ascending=False)

print(all_miss_data)
#Correlation matrix

Cor_heat=train_df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(Cor_heat,vmax=.8,square=True)
#Scatterplot



sns.set()

cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(train_df[cols],size=3)

plt.show()
#fill the houses with no pool

all_data['PoolQC']=all_data['PoolQC'].fillna('None')

#houses with no misc feature

all_data['MiscFeature']=all_data['MiscFeature'].fillna('None')

#houses with no Alley

all_data['Alley']=all_data['Alley'].fillna('None')

#houses with no fence

all_data['Fence']=all_data['Fence'].fillna('None')

#houses with no fire place

all_data['FireplaceQu']=all_data['FireplaceQu'].fillna('None')
all_data['LotFrontage']=all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#filling the four 'garage' categorical variabes with none and continuos variables with zero



for col in ('GarageFinish','GarageQual','GarageCond','GarageType'):

    all_data[col]=all_data[col].fillna('None')

    

for col in ('GarageYrBlt','GarageArea','GarageCars'):

    all_data[col]=all_data[col].fillna(0)
#filling Basement Categorical cariables with none and cuntinuous variables with Zero



for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):

    all_data[col]=all_data[col].fillna('None')

    

for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):

    all_data[col]=all_data[col].fillna(0)
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)

print(all_data['MSZoning'].value_counts())

all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
print(all_data['Utilities'].value_counts())

all_data=all_data.drop(['Utilities'],axis=1)
print(all_data['Functional'].value_counts())

all_data['Functional']=all_data['Functional'].fillna('Typical')
print(all_data['Electrical'].value_counts())

all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
print(all_data['KitchenQual'].value_counts())

all_data['KitchenQual']=all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
#replacing with mode values

all_data['Exterior1st']=all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd']=all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType']=all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#na in subclass means no class

print(all_data['MSSubClass'].value_counts())

all_data['MSSubClass']=all_data['MSSubClass'].fillna('None')
all_miss_data=((all_data.isnull().sum())/len(all_data))*100

all_miss_data=all_miss_data.drop(all_miss_data[all_miss_data==0].index).sort_values(ascending=False)

print(all_miss_data)
print(all_data.info())
#now convert series datatype to string datatype



all_data['MSSubClass']=all_data['MSSubClass'].astype(str)

#print(all_data['OverallQual'].value_counts())

all_data['OverallQual']=all_data['OverallQual'].astype(str)

all_data['OverallCond']=all_data['OverallCond'].astype(str)

all_data['YearBuilt']=all_data['YearBuilt'].astype(str)

all_data['YearRemodAdd']=all_data['YearRemodAdd'].astype(str)
#Label to encode the variable





from sklearn.preprocessing import LabelEncoder 

cols= ('FireplaceQu','BsmtQual','BsmtCond','GarageQual','GarageCond','ExterQual','ExterCond','HeatingQC','PoolQC','KitchenQual',

       'BsmtFinType1','BsmtFinType2','Functional','Fence','BsmtExposure','GarageFinish','LandSlope','LotShape','PavedDrive','Street',

       'Alley','CentralAir','MSSubClass','OverallCond','YrSold','MoSold')

for c in cols:

    lbl= LabelEncoder()

    lbl.fit(list(all_data[c].values))

    all_data[c]=lbl.transform(list(all_data[c].values))

    

#printing the shape

print('Shape all data:{}' .format(all_data.shape))
