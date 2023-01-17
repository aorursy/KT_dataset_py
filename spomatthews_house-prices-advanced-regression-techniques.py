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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.columns
#Visit the Label Values

train.SalePrice.describe()

#Oh my, look at the standard deviation!
import seaborn as sns
sns.distplot(train.SalePrice)

# Right-Skewed and Peaky
# We are going to need a robust model and some clever feature engineering
# And considering when these homes were sold... This is going to be a fun task
train.YrSold.unique()
# Let's start by getting rid of nulls
#Find and count missing values
train.isnull().sum().sort_values(ascending=False).head(20)

#Considering that there are 81 features and 1460 entries, this data is not in poor shape.
#Especially since it looks like many of these null values may just be because a home doesn't have a Garage, etc.
#Let's go one by one
##  **Handling Nulls**
#Pool Quality
train.PoolQC.unique()
train['PoolQC'] = train.PoolQC.fillna('No_Pool')

#New feature for home having a pool, the number of those who have one is too small for other data
train['HasPool']=0
train.loc[train['PoolArea']>0,'HasPool']=1
train = train.drop(['PoolArea','PoolQC'], axis=1)

#Misc Feature
train.MiscFeature.unique()
train['MiscFeature'] = train.MiscFeature.fillna('No_Misc_Feature')

#Alley
train.Alley.unique()
train['Alley'] = train.Alley.fillna('No_Alley_Access')

#Fence
train.Fence.unique()
train['Fence'] = train.Fence.fillna('No_Fence')

#Fireplace Quality
train.FireplaceQu.unique()
train['FireplaceQu'] = train.FireplaceQu.fillna('No_Fireplace')

#How many feet of street attached to lot aka Lot Frontage
train.LotFrontage.unique()
train['LotFrontage'] = train.LotFrontage.fillna(0)

#Garage Condition
train.GarageCond.unique()
train['GarageCond'] = train.GarageCond.fillna('No_Garage')

#Garage Type
train.GarageType.unique()
train['GarageType'] = train.GarageCond.fillna('No_Garage')

#Year Garage was built
train.GarageYrBlt.unique()
train['GarageYrBlt'] = train.GarageCond.fillna('No_Garage')

#Garage's Interior Finish
train.GarageFinish.unique()
train['GarageFinish'] = train.GarageFinish.fillna('No_Garage')

#Garage's Quality
train.GarageQual.unique()
train['GarageQual'] = train.GarageQual.fillna('No_Garage')

#New category for homes with no garage - may signify a type of home like condo
train['NoGarage'] = 0
train.loc[train['GarageQual']=='No_Garage','NoGarage']=1

#Basement Finish Type 2
train['BsmtFinType2'] = train.BsmtFinType2.fillna('No_Basement')

#Basment Exposure
train['BsmtExposure'] = train.BsmtExposure.fillna('No_Basement')

#Basement Condition
train['BsmtCond'] = train.BsmtCond.fillna('No_Basement')

#Basement Quality
train['BsmtQual'] = train.BsmtQual.fillna('No_Basement')

#Basement Finish Type 1
train['BsmtFinType1'] = train.BsmtFinType1.fillna('No_Basement')

#New category for homes with no basement - may signify a type of home like condo
train['No_Basement'] = 0
train.loc[train['BsmtQual']=='No_Basement','No_Basement'] = 1

#Masonry Veneer Area
train['MasVnrArea'] = train.MasVnrArea.fillna(0)

#Masonry Type
train['MasVnrType'] = train.MasVnrType.fillna('No_Veneer')

#Electrical
train.Electrical.unique()
train.Electrical.value_counts()
train['Electrical'] = train.Electrical.fillna('SBrkr')
# One Hot Encoding for categorical variables
categorical_vars = []
for x in train.columns:
    if train[x].dtype==object:
        categorical_vars.append(x)
train = train.join(pd.get_dummies(train[categorical_vars]))
train = train.drop(categorical_vars,axis=1)
#So now the purely categorical features are gone, right? Nope
train.head()


train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
import matplotlib.pyplot as plt
plt.scatter(train['TotalSF'],train['SalePrice'])
# This looks like a good feature - I wonder what's going on with those outliers, though - 8000 and 12000 SF but selling for 200k?
Incomplete_Homes = train.loc[train['SaleCondition_Partial']>0]
plt.scatter(Incomplete_Homes['OverallCond'],Incomplete_Homes['SalePrice'])
# Interesting... Nearly every home that shows as Incomplete also shows as OverallCondition of 5
# Maybe we should introduce this as a new category to increase the efficacy of OverallCond
# Hopefully we can remove some of these extraneous 5's and break it out into categories
train.OverallCond.hist()
train.loc[train['SaleCondition_Partial']>0,'OverallCond'] = 'Incomplete'
train.loc[train['OverallCond']==5].Id.count()
# What values from the dataset are numerical but should be categorical?
#MSSubClass, YrSold, MoSold, OverallCond, OverallQual
train = train.join(pd.get_dummies(train['MSSubClass'], prefix='SubClass'))
train = train.join(pd.get_dummies(train['YrSold'], prefix='YrSold'))
train = train.join(pd.get_dummies(train['MoSold'], prefix='Month'))
train = train.join(pd.get_dummies(train['OverallCond'],prefix='OverallCond'))
train = train.join(pd.get_dummies(train['OverallQual'],prefix='OverallQual'))
train = train.drop(['MSSubClass', 'MoSold', 'OverallCond', 'OverallQual'],axis=1)
#YrSold not dropped yet, so it can be used in the future for more features
plt.scatter(train['TotalSF'],train['SalePrice'])

# Let's try and see if we can map out what homes are on the younger side and what homes are on the older side
train.YearBuilt.mean() #1971
train.YearBuilt.std() #30

# 1 standard deviation outliers for newer and older homes
train['OldHome'] = 0
train.loc[train['YearBuilt']<1941, 'OldHome'] = 1
train['NewHome'] = 0
train.loc[train['YearBuilt']>2001,'NewHome'] = 1


#Try and find out of normal 
train.SalePrice.mean() #180921
train.SalePrice.std() #79442
# NoRidge, NridgHt, StoneBr, Timber let's check these out
train.loc[train['Neighborhood_NoRidge']==1].SalePrice.mean() #335295
train.loc[train['Neighborhood_NridgHt']==1].SalePrice.mean() #316270
train.loc[train['Neighborhood_StoneBr']==1].SalePrice.mean() #310499
train.loc[train['Neighborhood_Timber']==1].SalePrice.mean() #242247

train['PricePerSF'] = train['SalePrice']/train['TotalSF']
train.PricePerSF.mean() #69
train.PricePerSF.std() #15
#Outliers above 84
train.loc[train['Neighborhood_NoRidge']==1].PricePerSF.mean() #82.67
train.loc[train['Neighborhood_NridgHt']==1].PricePerSF.mean() #88.61
train.loc[train['Neighborhood_StoneBr']==1].PricePerSF.mean() #89.28
train.loc[train['Neighborhood_Timber']==1].PricePerSF.mean() #77.88

#Calculate relative age of the home, start at 1
train['RelativeAge'] = train['YrSold'] - train['YearRemodAdd'] + 1

#Outliers above 85
train.loc[(train['Neighborhood_NoRidge']==1) & (train['RelativeAge']<15)].PricePerSF.mean() #85.39
train.loc[(train['Neighborhood_NridgHt']==1) & (train['RelativeAge']<15)].PricePerSF.mean() #88.61
train.loc[(train['Neighborhood_StoneBr']==1) & (train['RelativeAge']<15)].PricePerSF.mean() #97.83
train.loc[(train['Neighborhood_Timber']==1) & (train['RelativeAge']<15)].PricePerSF.mean() #85

#These neighborhoods seem to have more premium homes, especially when built in the last few years
# Let's create a category for them: 'PremiumHome'

train['PremiumHome'] = 0
train.loc[((train['Neighborhood_NoRidge']==1) | (train['Neighborhood_NridgHt']==1) | (train['Neighborhood_StoneBr']==1) | (train['Neighborhood_Timber']==1))&(train['RelativeAge']<15), 'PremiumHome'] = 1
train = train.drop('YrSold', axis=1)
train = train.drop('PricePerSF',axis=1)
#Let's start Machine Learning
from sklearn.model_selection import train_test_split

y=train['SalePrice']
X= train.drop('SalePrice', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rfreg = RandomForestRegressor(n_estimators=500)

rfreg.fit(X_train, y_train)
rfreg.score(X_test, y_test)
#Now to retrofit the test data
test.isnull().sum().sort_values(ascending=False).head(20)
##  **Handling Nulls**
#Pool Quality
test.PoolQC.unique()
test['PoolQC'] = test.PoolQC.fillna('No_Pool')

#New feature for home having a pool, the number of those who have one is too small for other data
test['HasPool']=0
test.loc[test['PoolArea']>0,'HasPool']=1
test = test.drop(['PoolArea','PoolQC'], axis=1)

#Misc Feature
test.MiscFeature.unique()
test['MiscFeature'] = test.MiscFeature.fillna('No_Misc_Feature')

#Alley
test.Alley.unique()
test['Alley'] = test.Alley.fillna('No_Alley_Access')

#Fence
test.Fence.unique()
test['Fence'] = test.Fence.fillna('No_Fence')

#Fireplace Quality
test.FireplaceQu.unique()
test['FireplaceQu'] = test.FireplaceQu.fillna('No_Fireplace')

#How many feet of street attached to lot aka Lot Frontage
test.LotFrontage.unique()
test['LotFrontage'] = test.LotFrontage.fillna(0)

#Garage Condition
test.GarageCond.unique()
test['GarageCond'] = test.GarageCond.fillna('No_Garage')

#Garage Type
test.GarageType.unique()
test['GarageType'] = test.GarageCond.fillna('No_Garage')

#Year Garage was built
test.GarageYrBlt.unique()
test['GarageYrBlt'] = test.GarageCond.fillna('No_Garage')

#Garage's Interior Finish
test.GarageFinish.unique()
test['GarageFinish'] = test.GarageFinish.fillna('No_Garage')

#Garage's Quality
test.GarageQual.unique()
test['GarageQual'] = test.GarageQual.fillna('No_Garage')

#New category for homes with no garage - may signify a type of home like condo
test['NoGarage'] = 0
test.loc[test['GarageQual']=='No_Garage','NoGarage']=1

#Basement Finish Type 2
test['BsmtFinType2'] = test.BsmtFinType2.fillna('No_Basement')

#Basment Exposure
test['BsmtExposure'] = test.BsmtExposure.fillna('No_Basement')

#Basement Condition
test['BsmtCond'] = test.BsmtCond.fillna('No_Basement')

#Basement Quality
test['BsmtQual'] = test.BsmtQual.fillna('No_Basement')

#Basement Finish Type 1
test['BsmtFinType1'] = test.BsmtFinType1.fillna('No_Basement')

#New category for homes with no basement - may signify a type of home like condo
test['No_Basement'] = 0
test.loc[test['BsmtQual']=='No_Basement','No_Basement'] = 1

#Masonry Veneer Area
test['MasVnrArea'] = test.MasVnrArea.fillna(0)

#Masonry Type
test['MasVnrType'] = test.MasVnrType.fillna('No_Veneer')

#Electrical
test.Electrical.unique()
test.Electrical.value_counts()
test['Electrical'] = test.Electrical.fillna('SBrkr')

#MSZoning
test.MSZoning.unique()
test.MSZoning.value_counts()
test['MSZoning'] = test.MSZoning.fillna('RM')

#Functional
test.Functional.unique()
test.Functional.value_counts()
test['Functional'] = test.Functional.fillna('Typ')

#Utilities
test.Utilities.unique()
test['Utilities'] = test.Utilities.fillna('AllPub')

#BsmtHalfBath
test.BsmtHalfBath.value_counts()
test['BsmtHalfBath'] = test.BsmtHalfBath.fillna(0)

#BsmtFullBath
test.BsmtFullBath.value_counts()
test['BsmtFullBath'] = test.BsmtFullBath.fillna(0)

#KitchenQual
test.loc[test['KitchenQual'].isna(),'OverallQual']
test['KitchenQual'] = test.KitchenQual.fillna('TA')

#BsmtFinSF1, BsmtFinSF2, TotalBsmtSF, BsmtUnfSF
test.loc[test['BsmtFinSF1'].isna(), 'YearBuilt']
test['BsmtFinSF1'] = test.BsmtFinSF1.fillna(0)
test['BsmtFinSF2'] = test.BsmtFinSF2.fillna(0)
test['TotalBsmtSF'] = test.TotalBsmtSF.fillna(0)
test['BsmtUnfSF'] = test.BsmtUnfSF.fillna(0)

#Exterior1st
test.loc[test['YearBuilt']==1940].Exterior1st.value_counts()
test['Exterior1st'] = test.Exterior1st.fillna('Wd Sdng')
test['Exterior2nd'] = test.Exterior2nd.fillna('Wd Sdng')

#GarageCars, GarageArea
test.loc[test['GarageArea'].isna()]
test['GarageArea'] = test.GarageArea.fillna(0)
test['GarageCars'] = test.GarageCars.fillna(0)

#SaleType
test.SaleType.value_counts()
test['SaleType'] = test.SaleType.fillna('Typ')
#Now to retrofit the test data
test.isnull().sum().sort_values(ascending=False).head(20)
# One Hot Encoding for categorical variables
categorical_vars = []
for x in test.columns:
    if test[x].dtype==object:
        categorical_vars.append(x)
test = test.join(pd.get_dummies(test[categorical_vars]))
test = test.drop(categorical_vars,axis=1)

#Create TotalSF Feature
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

#Create Incomplete Homes Feature
test.loc[test['SaleCondition_Partial']>0,'OverallCond'] = 'Incomplete'
test
# One Hot Encoding for Test Numerical features
test = test.join(pd.get_dummies(test['MSSubClass'], prefix='SubClass'))
test = test.join(pd.get_dummies(test['YrSold'], prefix='YrSold'))
test = test.join(pd.get_dummies(test['MoSold'], prefix='Month'))
test = test.join(pd.get_dummies(test['OverallCond'],prefix='OverallCond'))
test = test.join(pd.get_dummies(test['OverallQual'],prefix='OverallQual'))
#Now Drop
test = test.drop(['MSSubClass', 'MoSold', 'OverallCond', 'OverallQual'],axis=1)
#Outlier Age Homes
test['OldHome'] = 0
test.loc[test['YearBuilt']<1941, 'OldHome'] = 1
test['NewHome'] = 0
test.loc[test['YearBuilt']>2001,'NewHome'] = 1
#Calculate relative age of the home, start at 1
test['RelativeAge'] = test['YrSold'] - test['YearRemodAdd'] + 1

#These neighborhoods seem to have more premium homes, especially when built in the last few years
# Let's create a category for them: 'PremiumHome'

test['PremiumHome'] = 0
test.loc[((test['Neighborhood_NoRidge']==1) | (test['Neighborhood_NridgHt']==1) | (test['Neighborhood_StoneBr']==1) | (test['Neighborhood_Timber']==1))&(test['RelativeAge']<15), 'PremiumHome'] = 1
#Final fitting of the test df
zerocols = ['Utilities_NoSeWa', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'HouseStyle_2.5Fin', 'RoofMatl_ClyTile', 'RoofMatl_Membran', 
'RoofMatl_Metal', 'RoofMatl_Roll', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other', 'Heating_Floor', 'Heating_OthW', 
'Electrical_Mix', 'GarageQual_Ex', 'MiscFeature_TenC']
for x in zerocols:
    test[x]=0

X['SaleType_Typ'] = 0
X['SubClass_150'] = 0

test = test.drop('YrSold', axis=1)

X = X.sort_index(axis=1)
test = test.sort_index(axis=1)
#Now let's start the ML
rfreg = RandomForestRegressor(n_estimators=500)

rfreg.fit(X,y)
submission = rfreg.predict(test)
submission = pd.DataFrame(submission)
submission.to_csv('HousePricesSubmission.csv')
