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
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train[:10]
train_id = train['Id']

test_id = test['Id']

train.drop('Id',inplace = True, axis = 1)

test.drop('Id',inplace = True, axis = 1)
train = train[train.GrLivArea < 4500]

train.reset_index(drop = True, inplace = True)

outliars = [30, 88, 462, 631, 1322]

train.drop(train.index[outliars], inplace = True)

#If you want to know why remove these values u can read kernels about EDA
#we don't need SalePrice so drop it

train.drop('SalePrice', inplace=True, axis=1)

data_features = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition'])).reset_index(drop=True)
#first get the total missing values

total = data_features.isnull().sum().sort_values(ascending=False)

total = total[total>0]

percent = total/len(data_features)

percent

missing_data = pd.concat((total,percent),axis=1,keys=['total','percent'])

missing_data
data_features.loc[(data_features['PoolQC'].notnull()),['PoolArea','PoolQC']]
#The code below just for a simple look

a = data_features.loc[data_features['MiscFeature'].notnull(),'MiscFeature']

data_features.loc[data_features['MiscFeature'].notnull(),'MiscFeature'].value_counts()
a = data_features.loc[data_features['Alley'].notnull(),'Alley']

data_features.loc[data_features['Alley'].notnull(),'Alley'].value_counts()
a = data_features.loc[data_features['Fence'].notnull(),'Fence']

data_features.loc[data_features['Fence'].notnull(),'Fence'].value_counts()
data_features['FireplaceQu'].groupby([data_features['Fireplaces'],data_features['FireplaceQu']]).count()
data_features.groupby('Neighborhood')['LotFrontage'].mean()
plt.figure(figsize=(16,9))

plt.plot(data_features['LotArea'], data_features['LotFrontage'])
x = data_features.loc[(data_features['LotArea'].notnull()), 'LotArea']

y = data_features.loc[(data_features['LotFrontage'].notnull()), 'LotFrontage']

t = (x<25000) & (y<150)

coef = np.polyfit(x[t], y[t],1)

formula = np.poly1d(coef)

poly_y = formula(data_features['LotArea'])

condition_frontage = (data_features['LotFrontage'].isnull())

data_features.loc[condition_frontage,'LotFrontage'] = formula(data_features.loc[condition_frontage,'LotArea'])

#use neighborhood to fill

#data_features['LotFrontage'] = data_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mode()[0]))
data_features['MSSubClass'].groupby((data_features['GarageCond'],data_features['GarageQual'])).count()

#These two variables have positive correlation.So we can use the mode to fill the missing in GarageCond and GarageQual. 
data_features.loc[(data_features['GarageCond'].isnull() & data_features['GarageQual'].notnull()), ['GarageQual']]

#it seems like they have the same missing values.

#the same for the two below

#so fill them all with NONE
data_features.loc[(data_features['GarageCond'].isnull() & data_features['GarageYrBlt'].notnull()), ['GarageYrBlt']]
data_features.loc[(data_features['GarageCond'].isnull() & data_features['GarageFinish'].notnull()), ['GarageFinish']]
data_features.loc[(data_features['GarageYrBlt'].isnull() & data_features['GarageType'].notnull())

                  ,['GarageYrBlt','GarageType','GarageCond','GarageFinish','GarageQual']]

#so we use the value of GarageType to fill the other four variables

garage_var = ['GarageYrBlt','GarageType','GarageCond','GarageFinish','GarageQual']

condition1 = (data_features['GarageYrBlt'].isnull() & data_features['GarageType'].notnull())

for col in garage_var:

    data_features.loc[condition1,col] = data_features[(data_features['GarageType'] == 'Detchd')][col].mode()[0]

#Note that we still have 156 missing values to fill for all 5 variables
#handle them in the same way as garage

missing_data[11:]
data_features[(data_features['BsmtFinType1'].isnull() & data_features['BsmtCond'].notnull())]

#We detect that all bsmt variables have 79 common missing values. Obviously these data mean there are no basements in these houses.

#So we use the NONE to fill them like the garage variables.
data_features[(data_features['BsmtFinType1'].notnull() & data_features['BsmtExposure'].isnull())][

    ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']]
data_features['MSSubClass'].groupby((data_features['BsmtQual'],data_features['BsmtExposure'])).count()
data_features['MSSubClass'].groupby((data_features['BsmtCond'],data_features['BsmtExposure'])).count()

#When Con is TA and Qual is Gd we shuold choose No to fill the missing value in 'BsmtExposure'
condition2 = (data_features['BsmtFinType1'].notnull() & data_features['BsmtExposure'].isnull())

data_features.loc[condition2,'BsmtExposure'] = 'No'
data_features[(data_features['BsmtFinType1'].notnull() & data_features['BsmtCond'].isnull())][

    ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']]

data_features['MSSubClass'].groupby((data_features['BsmtExposure'],data_features['BsmtCond'])).count()

#The proportion of TA is larger than other values so use TA to fill 'BsmtCond'
condition3 = (data_features['BsmtFinType1'].notnull() & data_features['BsmtCond'].isnull())

data_features.loc[condition3,'BsmtCond'] = 'TA'
data_features[(data_features['BsmtFinType1'].notnull() & data_features['BsmtQual'].isnull())][

    ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']]
data_features['MSSubClass'].groupby((data_features['BsmtExposure'],data_features['BsmtQual'])).count()
data_features['MSSubClass'].groupby((data_features['BsmtCond'],data_features['BsmtQual'])).count()
#Fill with TA again

condition4 = (data_features['BsmtFinType1'].notnull() & data_features['BsmtQual'].isnull())

data_features.loc[condition4,'BsmtQual'] = data_features.loc[(data_features['BsmtExposure'] == 'No'),'BsmtQual'].mode()[0]
#The last one 

condition5 = (data_features['BsmtFinType2'].isnull() & data_features['BsmtFinType1'].notnull())

data_features[condition5][['BsmtFinType1','BsmtFinType2']]

#From the data description 
data_features['MSSubClass'].groupby((data_features['BsmtFinType1'],data_features['BsmtFinType2'])).count()

#I guess even if Type1 is good , the Type 2 is more likely to be Unf. So fill 'BsmtFinType2' by Unf

data_features.loc[condition5, 'BsmtFinType2'] = 'Unf'
bsmt_var = ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']

garage_var = ['GarageType','GarageCond','GarageFinish','GarageQual']

NONE_var = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']

missing_data = data_features.isnull().sum().sort_values(ascending = False)

missing_data = missing_data[missing_data > 0]

missing_data
for col in bsmt_var, garage_var,NONE_var:

    data_features[col] = data_features[col].fillna('NONE')

data_features['GarageYrBlt'] = data_features['GarageYrBlt'].fillna(0)
data_features['LotFrontage']
missing_data = data_features.isnull().sum().sort_values(ascending = False)

missing_data = missing_data[missing_data > 0]

missing_data
condition6 = (data_features['MasVnrType'].isnull() & data_features['MasVnrArea'].notnull())

data_features.loc[condition6,['MasVnrType','MasVnrArea']]
data_features['MasVnrArea'].groupby(data_features['MasVnrType']).describe()
data_features['MasVnrArea'].groupby(data_features['MasVnrType']).median()
sns.boxplot(data_features['MasVnrType'],data_features['MasVnrArea'])

#Maybe fill it with 'Stone' is more reasonable.

#Something strange here. None means no masonry but there still several values. May theuy are outliars that I should remove?
a = data_features[(data_features['MasVnrType'] == 'None')]['MasVnrArea']

a[a>0]
data_features.loc[condition6,'MasVnrType'] = 'Stone'

data_features['MasVnrType'] = data_features['MasVnrType'].fillna('None')

data_features['MasVnrArea'] = data_features['MasVnrArea'].fillna(0)
data_features['MSZoning'].groupby([data_features['MSSubClass'],data_features['MSZoning']]).count()

data_features['MSZoning'] = data_features['MSZoning'].groupby(data_features['MSSubClass']).transform(lambda x:x.fillna(x.mode()[0]))
missing_data = data_features.isnull().sum().sort_values(ascending = False)

missing_data = missing_data[missing_data > 0]

missing_data
NA_for_0 = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',

            'GarageArea', 'GarageCars','MasVnrArea']

for col in NA_for_0:

    data_features[col] = data_features[col].fillna(0)
common_for_NA = ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']

for col in common_for_NA:

    data_features[col].fillna(data_features[col].mode()[0], inplace = True)
data_features['Functional'] = data_features['Functional'].fillna('Typ')

data_features['Utilities'] = data_features['Utilities'].fillna('None')
missing_data = data_features.isnull().sum().sort_values(ascending = False)

missing_data = missing_data[missing_data > 0]

missing_data
#思考到有三个缺失值不重合的的时候 应该怎么养去填充 或者说 bsmt相关的变量 应该以哪一个为基准去补充缺失值
data_features[(data_features['BsmtCond'].isnull() & data_features['BsmtFinType1'].notnull())][['BsmtQual','BsmtFinType1','BsmtFinType2']]
condition1 = (data_features['BsmtCond'].isnull() & data_features['BsmtExposure'].notnull())

data_features[condition1]