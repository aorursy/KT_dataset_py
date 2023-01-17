import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, RadioButtons

import seaborn as sns
#Reading both datasets

train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
# Numeric features

numeric_features=train.select_dtypes(include=[np.number])

numeric_features.dtypes

# Note MSSubClass,OverallQual and OverallCond are actually categorical features in numeric values.
# Categorical features

categoricals=train.select_dtypes(exclude=[np.number])

categoricals.dtypes
# Target variable

train.SalePrice.describe()
# Lets have a look at target/Y variable

plt.hist(train.SalePrice,color='blue')

plt.show()

#The data is skewed, hence to normalise it we will use log.
# Log of Y variable

targets=np.log(train.SalePrice)

plt.hist(targets,color='blue')

plt.show()

#Now the data is normal distributed, we will use log data for training the model .
#Lets combine both train and test datasets.

train_copy=train.copy()

train_copy.drop(['SalePrice'],axis=1,inplace=True)

combined=train_copy.append(test)

combined.reset_index(inplace=True)

combined.drop(['index','Id'],axis=1,inplace=True)
#Null values using heatmap

sns.heatmap(combined.isnull(), cbar=False)
# Correlation

corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending=False))
# Lets visualize correlation with heatmap

plt.subplots(figsize=(20, 9))

sns.heatmap(corr, square=True)
# YearBuilt and GarageYrBlt

plt.scatter(x=combined['YearBuilt'],y=combined['GarageYrBlt'])

plt.xlabel('YearBuilt')

plt.ylabel('GarageYrBlt')

plt.show()

#Note that in some cases YrBuilt is more than GarageYrBlt, which is impossible! 

#Hence we can assume that there was an error in filling the data. Also one point is clearly an outlier. 
#Quality vs SalePrice - High correlation

quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

quality_pivot.plot(kind='bar',color='blue')

plt.xlabel('OverallQual')

plt.ylabel('SalePrice')

plt.show()
#OverallCond vs SalePrice - Not much correlation

cond_pivot = train.pivot_table(index='OverallCond', values='SalePrice', aggfunc=np.median)

cond_pivot.plot(kind='bar',color='blue')

plt.xlabel('OverallCond')

plt.ylabel('SalePrice')

plt.show()
#HalfBath vs SalePrice

halfbath_pivot = train.pivot_table(index='HalfBath', values='SalePrice', aggfunc=np.median)

halfbath_pivot.plot(kind='bar',color='blue')

plt.xlabel('HalfBath')

plt.ylabel('SalePrice')

plt.show()
#FullBath vs SalePrice

fullbath_pivot = train.pivot_table(index='FullBath', values='SalePrice', aggfunc=np.median)

fullbath_pivot.plot(kind='bar',color='blue')

plt.xlabel('FullBath')

plt.ylabel('SalePrice')

plt.show()
#BsmtFullBath vs SalePrice

BsmtFullBath_pivot = train.pivot_table(index='BsmtFullBath', values='SalePrice', aggfunc=np.median)

BsmtFullBath_pivot.plot(kind='bar',color='blue')

plt.xlabel('BsmtFullBath')

plt.ylabel('SalePrice')

plt.show()
#BsmtHalfBath vs SalePrice

BsmtHalfBath_pivot = train.pivot_table(index='BsmtHalfBath', values='SalePrice', aggfunc=np.median)

BsmtHalfBath_pivot.plot(kind='bar',color='blue')

plt.xlabel('BsmtHalfBath')

plt.ylabel('SalePrice')

plt.show()
#BedroomAbvGr vs SalesPrice

BedroomAbvGr_pivot = train.pivot_table(index='BedroomAbvGr', values='SalePrice', aggfunc=np.median)

BedroomAbvGr_pivot.plot(kind='bar',color='blue')

plt.xlabel('BedroomAbvGr')

plt.ylabel('SalePrice')

plt.show()
#Fireplaces vs SalePrice

Fireplaces_pivot = train.pivot_table(index='Fireplaces', values='SalePrice', aggfunc=np.median)

Fireplaces_pivot.plot(kind='bar',color='blue')

plt.xlabel('Fireplaces')

plt.ylabel('SalePrice')

plt.show()
#TotRmsAbvGrd vs SalePrice

TotRmsAbvGrd_pivot = train.pivot_table(index='TotRmsAbvGrd', values='SalePrice', aggfunc=np.median)

TotRmsAbvGrd_pivot.plot(kind='bar',color='blue')

plt.xlabel('TotRmsAbvGrd')

plt.ylabel('SalePrice')

plt.show()
#KitchenAbvGr vs SalePrice

KitchenAbvGr_pivot = train.pivot_table(index='KitchenAbvGr', values='SalePrice', aggfunc=np.median)

KitchenAbvGr_pivot.plot(kind='bar',color='blue')

plt.xlabel('KitchenAbvGr')

plt.ylabel('SalePrice')

plt.show()
#GrLivArea Vs SalePrice

plt.scatter(x=train['GrLivArea'],y=targets)

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()
#GarageCars vs SalePrice

plt.scatter(x=train['GarageCars'],y=targets)

plt.xlabel('GarageCars')

plt.ylabel('SalePrice')

plt.show()
#GarageArea vs SalePrice

plt.scatter(x=train['GarageArea'],y=targets)

plt.xlabel('GarageArea')

plt.ylabel('SalePrice')

plt.show()
# Numeric variables vs SalePrice



sns.set_style('darkgrid')

plt.figure(figsize=(26,32))

plt.subplot(17,2,1)

plt.tight_layout()

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

plt.hist(combined['LotFrontage'], bins=30, edgecolor= 'black',color ='teal')

plt.title('LotFrontage')



plt.subplot(17,2,2)

plt.scatter(x=train.LotFrontage, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('LotFrontage vs SalePrice')



plt.subplot(17,2,3)

plt.hist(combined['MasVnrArea'], bins=30, edgecolor= 'black',color ='teal')

plt.title('MasVnrArea')



plt.subplot(17,2,4)

plt.scatter(x=train.MasVnrArea, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('MasVnrArea vs SalePrice')



plt.subplot(17,2,5)

plt.hist(combined['LotArea'], bins=30, edgecolor= 'black',color ='teal')

plt.title('LotArea')



plt.subplot(17,2,6)

plt.scatter(x=train.LotArea, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('LotArea vs SalePrice')



plt.subplot(17,2,7)

plt.hist(combined['BsmtFinSF1'], bins=30, edgecolor= 'black',color ='teal')

plt.title('BsmtFinSF1')



plt.subplot(17,2,8)

plt.scatter(x=train.BsmtFinSF1, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('BsmtFinSF1 vs SalePrice')



plt.subplot(17,2,9)

plt.hist(combined['BsmtFinSF2'], bins=30, edgecolor= 'black',color ='teal')

plt.title('BsmtFinSF2')



plt.subplot(17,2,10)

plt.scatter(x=train.BsmtFinSF2, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('BsmtFinSF2 vs SalePrice')



plt.subplot(17,2,11)

plt.hist(combined['BsmtUnfSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('BsmtUnfSF')



plt.subplot(17,2,12)

plt.scatter(x=train.BsmtUnfSF, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('BsmtUnfSF vs SalePrice')



plt.subplot(17,2,13)

plt.hist(combined['TotalBsmtSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('TotalBsmtSF')



plt.subplot(17,2,14)

plt.scatter(x=train.TotalBsmtSF, y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('TotalBsmtSF vs SalePrice')



plt.subplot(17,2,15)

plt.hist(combined['1stFlrSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('1stFlrSF')



plt.subplot(17,2,16)

plt.scatter(x=train['1stFlrSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('1stFlrSF vs SalePrice')



plt.subplot(17,2,17)

plt.hist(combined['2ndFlrSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('2ndFlrSF')



plt.subplot(17,2,18)

plt.scatter(x=train['2ndFlrSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('2ndFlrSF vs SalePrice')



plt.subplot(17,2,19)

plt.hist(combined['LowQualFinSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('LowQualFinSF')



plt.subplot(17,2,20)

plt.scatter(x=train['LowQualFinSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('LowQualFinSF vs SalePrice')



plt.subplot(17,2,21)

plt.hist(combined['WoodDeckSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('WoodDeckSF')



plt.subplot(17,2,22)

plt.scatter(x=train['WoodDeckSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('WoodDeckSF vs SalePrice')



plt.subplot(17,2,23)

plt.hist(combined['OpenPorchSF'], bins=30, edgecolor= 'black',color ='teal')

plt.title('OpenPorchSF')



plt.subplot(17,2,24)

plt.scatter(x=train['OpenPorchSF'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('OpenPorchSF vs SalePrice')



plt.subplot(17,2,25)

plt.hist(combined['EnclosedPorch'], bins=30, edgecolor= 'black',color ='teal')

plt.title('EnclosedPorch')



plt.subplot(17,2,26)

plt.scatter(x=train['EnclosedPorch'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('EnclosedPorch vs SalePrice')



plt.subplot(17,2,27)

plt.hist(combined['3SsnPorch'], bins=30, edgecolor= 'black',color ='teal')

plt.title('3SsnPorch')



plt.subplot(17,2,28)

plt.scatter(x=train['3SsnPorch'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('3SsnPorch vs SalePrice')



plt.subplot(17,2,29)

plt.hist(combined['ScreenPorch'], bins=30, edgecolor= 'black',color ='teal')

plt.title('ScreenPorch')



plt.subplot(17,2,30)

plt.scatter(x=train['ScreenPorch'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('ScreenPorch vs SalePrice')



plt.subplot(17,2,31)

plt.hist(combined['PoolArea'], bins=30, edgecolor= 'black',color ='teal')

plt.title('PoolArea')



plt.subplot(17,2,32)

plt.scatter(x=train['PoolArea'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('PoolArea vs SalePrice')



plt.subplot(17,2,33)

plt.hist(combined['MiscVal'], bins=30, edgecolor= 'black',color ='teal')

plt.title('MiscVal')



plt.subplot(17,2,34)

plt.scatter(x=train['MiscVal'], y=train.SalePrice,edgecolor= 'black',color ='teal')

plt.title('MiscVal vs SalePrice')



plt.show()
#MSSubClass

sns.countplot(combined.MSSubClass)

plt.show()

sns.boxplot(x = 'MSSubClass', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MSSubclass')

#Note correlation seems to be very less with Sale Price
#MSZoning

sns.countplot(combined.MSZoning)

plt.show()

sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MSZoning')
#Street

sns.countplot(combined.Street)

plt.show()

sns.violinplot(x = 'Street', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Street')
#Utilities

sns.countplot(combined.Utilities) 

#Entire column can be deleted as it contains only one level
#Foundation

sns.countplot(combined.Foundation)

sns.violinplot(x = 'Foundation', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Foundation')
#BsmtCond

sns.countplot(combined.BsmtCond)

plt.show()

sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtCond')
#BsmtQual

sns.countplot(combined.BsmtQual)

plt.show()

sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtCond')
#BsmtFinType1

sns.countplot(combined.BsmtFinType1)

plt.show()

sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtFinType1')
#BsmtFinType2

sns.countplot(combined.BsmtFinType2)

plt.show()

sns.boxplot(x = 'BsmtFinType2', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BsmtFinType2')

#Majority is unfinished with clearly no impact on Sale Price
#LotShape 

sns.countplot(combined.LotShape)

plt.show()

sns.boxplot(x = 'LotShape', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LotShape')
#Alley - Can conclude that paved alley leads to much higher prices than gravel

sns.countplot(combined.Alley)

plt.show()

sns.boxplot(x = 'Alley', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Alley')
#ExterQual

sns.countplot(combined.ExterQual)

plt.show()

sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by ExterQual')
#ExterCond

sns.countplot(combined.ExterCond)

plt.show()

sns.boxplot(x = 'ExterCond', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by ExterCond')

#Note most are TA, hardly any correlation in the data
#Heating

sns.countplot(combined.Heating)

plt.show()

sns.boxplot(x = 'Heating', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Heating')

#Data dominated by one category(GasA)
#HeatingQC

sns.countplot(combined.HeatingQC)

plt.show()

sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by HeatingQC')
#CentralAir

sns.countplot(combined.CentralAir)

plt.show()

sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by CentralAC')

#With CentralAC saleprices are high compared to NonAC
#FireplaceQu

sns.countplot(combined.FireplaceQu)

plt.show()

sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by FireplaceQu')
#GarageType

sns.countplot(combined.GarageType)

plt.show()

sns.boxplot(x = 'GarageType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by GarageType')
#GarageFinish

sns.countplot(combined.GarageFinish)

plt.show()

sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by GarageFinish')
#GarageQual

sns.countplot(combined.GarageQual)

plt.show()

sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by GarageQual')
#Fence

sns.countplot(combined.Fence)

plt.show()

sns.boxplot(x = 'Fence', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Fence')
#PavedDrive

sns.countplot(combined.PavedDrive)

plt.show()

sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by PavedDrive')
#LandSlope

sns.countplot(combined.LandSlope)

plt.show()

sns.boxplot(x = 'LandSlope', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LandSlope')

#Not much corelation here!
#Condition1

sns.countplot(combined.Condition1)

plt.show()

sns.boxplot(x = 'Condition1', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Condition1')
#Condition2

sns.countplot(combined.Condition2)

plt.show()

sns.boxplot(x = 'Condition2', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Condition2')
#Functional

sns.countplot(combined.Functional)

plt.show()

sns.boxplot(x = 'Functional', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Fuctional')
#KitchenQual

sns.countplot(combined.KitchenQual)

plt.show()

sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by KitchenQual')
#RoofMatl

sns.countplot(combined.RoofMatl)

plt.show()

sns.violinplot(x = 'RoofMatl', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by RoofMatl')

#Important indicator of SalePrice
#RoofStyle

sns.countplot(combined.RoofStyle)

plt.show()

sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by RoofStyle')
#Electrical

sns.countplot(combined.Electrical)

plt.show()

sns.boxplot(x = 'Electrical', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Electrical')
#Exterior1st

sns.countplot(combined.Exterior1st)

plt.show()

sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Exterior1st')
#Exterior2nd

sns.countplot(combined.Exterior2nd)

plt.show()

sns.boxplot(x = 'Exterior2nd', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by Exterior2nd')
#BldgType

sns.countplot(combined.BldgType) 

plt.show()

sns.boxplot(x = 'BldgType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by BldgType')
#HouseStyle

sns.countplot(combined.HouseStyle)

plt.show()

sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by HouseStyle')
#LotConfig

sns.countplot(combined.LotConfig)

plt.show()

sns.boxplot(x = 'LotConfig', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LotConfig')
#SaleType

sns.countplot(combined.SaleType)

plt.show()

sns.boxplot(x = 'SaleType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by SaleType')
#SaleCondition

sns.countplot(combined.SaleCondition)

plt.show()

sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by SaleCondition')
#MasVnrType

sns.countplot(combined.MasVnrType)

plt.show()

sns.boxplot(x = 'MasVnrType', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MasVnrType')
#MiscFeature

sns.countplot(combined.MiscFeature)

plt.show()

sns.boxplot(x = 'MiscFeature', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by MiscFeature')
#LandContour

sns.countplot(combined.LandContour)

plt.show()

sns.boxplot(x = 'LandContour', y = 'SalePrice', data = train, palette= 'GnBu_d').set_title('Sale Price by LandContour')
#Neighborhood

sns.countplot(combined.Neighborhood)

plt.xticks(rotation=45)

plt.show()

sns.stripplot(x = train.Neighborhood.values, y = train.SalePrice,

              order = np.sort(train.Neighborhood.unique()),

              jitter=0.1, alpha=0.5).set_title('Sale Price by Neighbourhood')

 

plt.xticks(rotation=45)