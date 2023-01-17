import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns       
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import sklearn
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("../input/train.csv",index_col="Id")
test = pd.read_csv("../input/test.csv",index_col="Id")
print(train.shape)
print(test.shape)
# SalePrice  房价
train['SalePrice'].value_counts()
train['SalePrice'].isnull().sum()
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
train['totalArea'] = train['GrLivArea'] + train['GarageArea'] + train['TotalBsmtSF']
train = train.drop(train[train['totalArea'] > 8000].index)
train['totalArea'] = np.log1p(train['totalArea'])
sns.distplot(train['totalArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['totalArea'], plot=plt)
fig = plt.figure()
plt.plot(train['totalArea'],train['SalePrice'],'o')

#test
test['GrLivArea'] = test['GrLivArea'].fillna(0)
test['GarageArea'] = test['GarageArea'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['totalArea'] = test['GrLivArea'] + test['GarageArea'] + test['TotalBsmtSF']
test['totalArea'] = np.log1p(test['totalArea'])
# SaleCondition 销售类型
# print(train['SaleCondition'].value_counts())
# train['SaleCondition'].isnull().sum()
# train = train[train['SaleCondition'] == 'Normal']
train = train.drop(train[train['GrLivArea'] > 4000].index)
plt.plot(train['GrLivArea'][train['SaleCondition'] == 'Normal'],train['SalePrice'][train['SaleCondition'] == 'Normal'],'o')
plt.plot(train['GrLivArea'][train['SaleCondition'] == 'Partial'],train['SalePrice'][train['SaleCondition'] == 'Partial'],'o')
plt.plot(train['GrLivArea'][train['SaleCondition'] == 'Abnorml'],train['SalePrice'][train['SaleCondition'] == 'Abnorml'],'o')
plt.plot(train['GrLivArea'][train['SaleCondition'] == 'Family'],train['SalePrice'][train['SaleCondition'] == 'Family'],'o')
plt.plot(train['GrLivArea'][train['SaleCondition'] == 'Alloca'],train['SalePrice'][train['SaleCondition'] == 'Alloca'],'o')
plt.plot(train['GrLivArea'][train['SaleCondition'] == 'AdjLand'],train['SalePrice'][train['SaleCondition'] == 'AdjLand'],'o')

# MSSubClass 建筑的类别
# print(train['MSZoning'].value_counts())
# train['SaleCondition'].isnull().sum()
train = train.drop(train[(train['GrLivArea'] > 3000) & (train['MSZoning'] == 'RM')].index)
train = train.drop(train[(train['GrLivArea'] > 3000) & (train['MSZoning'] == 'RH')].index)
plt.plot(train['GrLivArea'][train['MSZoning'] == 'RL'],train['SalePrice'][train['MSZoning'] == 'RL'],'o')
plt.plot(train['GrLivArea'][train['MSZoning'] == 'RM'],train['SalePrice'][train['MSZoning'] == 'RM'],'o')
plt.plot(train['GrLivArea'][train['MSZoning'] == 'FV'],train['SalePrice'][train['MSZoning'] == 'FV'],'o')
plt.plot(train['GrLivArea'][train['MSZoning'] == 'RH'],train['SalePrice'][train['MSZoning'] == 'RH'],'o')
plt.plot(train['GrLivArea'][train['MSZoning'] == 'C (all)'],train['SalePrice'][train['MSZoning'] == 'C (all)'],'o')

#test
test['MSZoning'] = test['MSZoning'].fillna('RL')
# LotFrontage 临街
# print(train['LotFrontage'].value_counts().sort_index())
train = train.drop(train[train['LotFrontage'] > 300].index)
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['not_LotFrontage'] = pd.Series(np.zeros((len(train))),index = train.index)
train['not_LotFrontage'][train['LotFrontage'] < 25] = 1
index = train['LotFrontage'] > 25

train['LotFrontage'] = np.log1p(train['LotFrontage'])
# # train['LotFrontage'].isnull().sum()
sns.distplot(train['LotFrontage'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['LotFrontage'][index], plot=plt)

# test
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['not_LotFrontage'] = pd.Series(np.zeros((len(test))),index = test.index)
test['not_LotFrontage'][test['LotFrontage'] < 25] = 1
test['LotFrontage'] = np.log1p(test['LotFrontage'])
# LotArea
# print(train['LotArea'].value_counts().sort_index())
train = train.drop(train[train['LotArea'] > 100000].index)
train['LotArea'] = np.log1p(train['LotArea'])
sns.distplot(train['LotArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['LotArea'], plot=plt)
fig = plt.figure()
plt.plot(train['LotArea'],train['SalePrice'],'o')

# test
test['LotArea'] = np.log1p(test['LotArea'])
# Street Alley 街道 胡同
train = train.drop('Street', axis=1)
train = train.drop('Alley', axis=1)

#test
test = test.drop('Street', axis=1)
test = test.drop('Alley', axis=1)
# LotShape 房屋的大致形状
# print(train['LotShape'].value_counts().sort_index())
# print(train['LotShape'].isnull().sum()/len(train))
plt.plot(train['GrLivArea'][train['LotShape'] == 'IR1'],train['SalePrice'][train['LotShape'] == 'IR1'],'o')
plt.plot(train['GrLivArea'][train['LotShape'] == 'Reg'],train['SalePrice'][train['LotShape'] == 'Reg'],'o')
plt.plot(train['GrLivArea'][train['LotShape'] == 'IR2'],train['SalePrice'][train['LotShape'] == 'IR2'],'o')
plt.plot(train['GrLivArea'][train['LotShape'] == 'IR3'],train['SalePrice'][train['LotShape'] == 'IR3'],'o')
# LandContour 物业
# print(train['LandContour'].value_counts().sort_index())
# print(train['LandContour'].isnull().sum()/len(train))
plt.plot(train['GrLivArea'][train['LandContour'] == 'Lvl'],train['SalePrice'][train['LandContour'] == 'Lvl'],'o')
plt.plot(train['GrLivArea'][train['LandContour'] == 'Bnk'],train['SalePrice'][train['LandContour'] == 'Bnk'],'o')
plt.plot(train['GrLivArea'][train['LandContour'] == 'HLS'],train['SalePrice'][train['LandContour'] == 'HLS'],'o')
plt.plot(train['GrLivArea'][train['LandContour'] == 'Low'],train['SalePrice'][train['LandContour'] == 'Low'],'o')
# Utilities 公共设施可用度
# print(train['Utilities'].value_counts().sort_index())
# print(train['Utilities'].isnull().sum()/len(train))
train = train.drop('Utilities', axis=1)

#test
test = test.drop('Utilities', axis=1)
# LotConfig
# print(train['LotConfig'].value_counts().sort_index())
# print(train['LotConfig'].isnull().sum()/len(train))
plt.plot(train['GrLivArea'][train['LotConfig'] == 'Corner'],train['SalePrice'][train['LotConfig'] == 'Corner'],'o')
plt.plot(train['GrLivArea'][train['LotConfig'] == 'Inside'],train['SalePrice'][train['LotConfig'] == 'Inside'],'o')
plt.plot(train['GrLivArea'][train['LotConfig'] == 'CulDSac'],train['SalePrice'][train['LotConfig'] == 'CulDSac'],'o')
plt.plot(train['GrLivArea'][train['LotConfig'] == 'FR3'],train['SalePrice'][train['LotConfig'] == 'FR3'],'o')
# LandSlope 坡度
# print(train['LandSlope'].value_counts().sort_index())
# print(train['LandSlope'].isnull().sum()/len(train))
plt.plot(train['GrLivArea'][train['LandSlope'] == 'Gtl'],train['SalePrice'][train['LandSlope'] == 'Gtl'],'o')
plt.plot(train['GrLivArea'][train['LandSlope'] == 'Mod'],train['SalePrice'][train['LandSlope'] == 'Mod'],'o')
plt.plot(train['GrLivArea'][train['LandSlope'] == 'Sev'],train['SalePrice'][train['LandSlope'] == 'Sev'],'o')
# Neighborhood  Condition1 Condition2位置
plt.plot(train['GrLivArea'][train['Condition1'] == 'Norm'],train['SalePrice'][train['Condition1'] == 'Norm'],'o')
plt.plot(train['GrLivArea'][train['Condition1'] == 'Artery'],train['SalePrice'][train['Condition1'] == 'Artery'],'o')
plt.plot(train['GrLivArea'][train['Condition1'] == 'Feedr'],train['SalePrice'][train['Condition1'] == 'Feedr'],'o')
# BldgType 住宅风格
# print(train['OverallQual'].value_counts().sort_index())
# print(train['OverallQual'].isnull().sum()/len(train))
sns.distplot(train['OverallQual'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['OverallQual'], plot=plt)
fig = plt.figure()
plt.plot(train['OverallQual'],train['SalePrice'],'o')
# OverallCond 总体状况评级
# print(train['OverallCond'].value_counts().sort_index())
# print(train['OverallCond'].isnull().sum()/len(train))
train = train.drop(train[(train['OverallCond'] ==2 )&(train['SalePrice'] > 300000 )].index)
sns.distplot(train['OverallCond'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['OverallCond'], plot=plt)
fig = plt.figure()
plt.plot(train['OverallCond'],train['SalePrice'],'o')
# YearBuilt
train['newhouse']  = pd.Series(np.zeros((len(train))),index = train.index)
train['newhouse'][train['YearBuilt'] > 2000]  = 1
train['age']  = 2010 - train['YearBuilt']
# print(train['age'].value_counts().sort_index())
# print(train['YearBuilt'].value_counts().sort_index())
# print(train['YearBuilt'].isnull().sum()/len(train))
plt.plot(train['YearBuilt'],train['SalePrice'],'o')

# test
test['newhouse']  = pd.Series(np.zeros((len(test))),index = test.index)
test['newhouse'][test['YearBuilt'] > 2000]  = 1
test['age']  = 2010 - test['YearBuilt']
# YearRemodAdd
# print(train['YearRemodAdd'].value_counts().sort_index())
# print(train['YearRemodAdd'].isnull().sum()/len(train))
train['1950house']  = pd.Series(np.zeros((len(train))),index = train.index)
train['1950house'][train['YearRemodAdd'] == 1950]  = 1
sns.distplot(train['YearRemodAdd'], fit=norm)

#test
test['1950house']  = pd.Series(np.zeros((len(test))),index = test.index)
test['1950house'][test['YearRemodAdd'] == 1950]  = 1
# Foundation
# print(train['Foundation'].value_counts().sort_index())
# print(train['Foundation'].isnull().sum()/len(train))
plt.plot(train['GrLivArea'][train['Foundation'] == 'BrkTil'],train['SalePrice'][train['Foundation'] == 'BrkTil'],'o')
plt.plot(train['GrLivArea'][train['Foundation'] == 'CBlock'],train['SalePrice'][train['Foundation'] == 'CBlock'],'o')
plt.plot(train['GrLivArea'][train['Foundation'] == 'PConc'],train['SalePrice'][train['Foundation'] == 'PConc'],'o')
# BsmtQual 地下室高度
train['BsmtQual'] = train['BsmtQual'].fillna("unknown")
train['BsmtCond'] = train['BsmtCond'].fillna("TA")
train['BsmtExposure'] = train['BsmtExposure'].fillna("unknown")
train['BsmtFinType1'] = train['BsmtFinType1'].fillna("unknown")

#test
test['BsmtQual'] = test['BsmtQual'].fillna("unknown")
test['BsmtCond'] = test['BsmtCond'].fillna("TA")
test['BsmtExposure'] = test['BsmtExposure'].fillna("unknown")
test['BsmtFinType1'] = test['BsmtFinType1'].fillna("unknown")
# print(train['BsmtFinSF1'].value_counts())
# print(train['BsmtFinSF1'].isnull().sum()/len(train))
train['noSF1']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noSF1'][train['BsmtFinSF1'] == 0]  = 1
plt.plot(train['BsmtFinSF1'],train['SalePrice'],'o')

# test
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
test['noSF1']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noSF1'][test['BsmtFinSF1'] == 0]  = 1
# BsmtCond
train['BsmtCond'] = train['BsmtCond'].fillna(0)

#test 
test['BsmtCond'] = test['BsmtCond'].fillna(0)
# print(train['BsmtUnfSF'].value_counts())
# print(train['TotalBsmtSF'].isnull().sum()/len(train))
train['noTotalBsmtSF']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noTotalBsmtSF'][train['TotalBsmtSF'] == 0]  = 1
index = train['TotalBsmtSF'] > 0
train['TotalBsmtSF'] = np.log1p(train['TotalBsmtSF'])
sns.distplot(train['TotalBsmtSF'][index], fit=norm)
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'][index], plot=plt)
fig = plt.figure()
plt.plot(train['TotalBsmtSF'][index],train['SalePrice'][index],'o')

#test
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['noTotalBsmtSF']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noTotalBsmtSF'][test['TotalBsmtSF'] == 0]  = 1
test['TotalBsmtSF'] = np.log1p(test['TotalBsmtSF'])


# Heating
train['Electrical'] = train['Electrical'].fillna('SBrkr')

# test
test['Electrical'] = test['Electrical'].fillna('SBrkr')
# print(train['1stFlrSF'].value_counts())
# print(train['1stFlrSF'].isnull().sum()/len(train))
train['1stFlrSF'] = np.log1p(train['1stFlrSF'])
sns.distplot(train['1stFlrSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['1stFlrSF'], plot=plt)

# test
test['1stFlrSF'] = np.log1p(test['1stFlrSF'])
# print(train['2ndFlrSF'].value_counts())
# print(train['2ndFlrSF'].isnull().sum()/len(train))
train['no2ndFlrSF']  = pd.Series(np.zeros((len(train))),index = train.index)
train['no2ndFlrSF'][train['2ndFlrSF'] == 0]  = 1
sns.distplot(train['2ndFlrSF'], fit=norm)

# test
test['no2ndFlrSF']  = pd.Series(np.zeros((len(test))),index = test.index)
test['no2ndFlrSF'][test['2ndFlrSF'] == 0]  = 1
#LowQualFinSF
# print(train['LowQualFinSF'].value_counts())
# print(train['LowQualFinSF'].isnull().sum()/len(train))
train['haveLowQual']  = pd.Series(np.zeros((len(train))),index = train.index)
train['haveLowQual'][train['LowQualFinSF'] > 0]  = 1

# test
test['haveLowQual']  = pd.Series(np.zeros((len(test))),index = test.index)
test['haveLowQual'][test['LowQualFinSF'] > 0]  = 1
# GrLivArea
# print(train['LowQualFinSF'].value_counts())
# print(train['LowQualFinSF'].isnull().sum()/len(train))

train['GrLivArea'] = np.log1p(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

#test
test['GrLivArea'] = np.log1p(test['GrLivArea'])
# FullBath
# print(train['FullBath'].value_counts())
# print(train['FullBath'].isnull().sum()/len(train))
sns.distplot(train['FullBath'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['FullBath'], plot=plt)
fig = plt.figure()
plt.plot(train['FullBath'],train['SalePrice'],'o')
# BsmtFullBath
# train.columns
# print(train['TotRmsAbvGrd'].value_counts())
# print(train['TotRmsAbvGrd'].isnull().sum()/len(train))
train['TotRmsAbvGrd'] = np.log1p(train['TotRmsAbvGrd'])
sns.distplot(train['TotRmsAbvGrd'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotRmsAbvGrd'], plot=plt)

# test
test['TotRmsAbvGrd'] = test['TotRmsAbvGrd'].fillna(test['TotRmsAbvGrd'].mean())
test['TotRmsAbvGrd'] = np.log1p(test['TotRmsAbvGrd'])
# Functional
# print(train['Fireplaces'].value_counts())
# print(train['Fireplaces'].isnull().sum()/len(train))
train = train.drop('FireplaceQu',axis=1)

# test
test = test.drop('FireplaceQu',axis=1)
# print(train['GarageType'].value_counts())
# print(train['GarageType'].isnull().sum()/len(train))
train['GarageType'] = train['GarageType'].fillna('unknown')

# test
test['GarageType'] = test['GarageType'].fillna('unknown')
# GarageYrBlt
# print(train['GarageYrBlt'].value_counts())
# print(train['GarageYrBlt'].isnull().sum()/len(train))
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
# plt.plot(train['GarageYrBlt'],train['SalePrice'],'o')

# test
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())
# GarageFinish
# print(train['GarageFinish'].value_counts())
# print(train['GarageFinish'].isnull().sum()/len(train))
train['GarageFinish'] = train['GarageFinish'].fillna('unknown')

# test
test['GarageFinish'] = test['GarageFinish'].fillna('unknown')
# print(train['GarageCars'].value_counts())
# print(train['GarageCars'].isnull().sum()/len(train))
# train['GarageCars'] = np.log1p(train['GarageCars'])
sns.distplot(train['GarageCars'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GarageCars'], plot=plt)

#test
test['GarageCars'] = test['GarageCars'].fillna(0)
# GarageArea
# print(train['GarageArea'].value_counts())
# print(train['GarageArea'].isnull().sum()/len(train))
train['noGarageArea']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noGarageArea'][train['GarageArea'] == 0]  = 1
index = train['GarageArea'] > 0
train['GarageArea'] = np.log1p(train['GarageArea'])
sns.distplot(train['GarageArea'][index], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GarageArea'][index], plot=plt)

#test
test['GarageArea'] = test['GarageArea'].fillna(0)
test['noGarageArea']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noGarageArea'][test['GarageArea'] == 0]  = 1
test['GarageArea'] = np.log1p(test['GarageArea'])
# GarageQual
# print(train['GarageQual'].value_counts())
# print(train['GarageQual'].isnull().sum()/len(train))
train['GarageQual'] = train['GarageQual'].fillna('unknown')

#test
test['GarageQual'] = test['GarageQual'].fillna('unknown')
# GarageCond
# print(train['GarageCond'].value_counts())
# print(train['GarageCond'].isnull().sum()/len(train))
train['GarageCond'] = train['GarageCond'].fillna('unknown')

#test
test['GarageCond'] = test['GarageCond'].fillna('unknown')
# WoodDeckSF
train['noOpenPorchSF']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noOpenPorchSF'][train['OpenPorchSF'] == 0]  = 1

index = train['OpenPorchSF'] > 0
train['OpenPorchSF'] = np.log1p(train['OpenPorchSF'])

sns.distplot(train['OpenPorchSF'][index], fit=norm);
fig = plt.figure()
res = stats.probplot(train['OpenPorchSF'][index], plot=plt)

#test
test['noOpenPorchSF']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noOpenPorchSF'][test['OpenPorchSF'] == 0]  = 1
test['OpenPorchSF'] = np.log1p(test['OpenPorchSF'])
# EnclosedPorch
# print(train['EnclosedPorch'].value_counts())
train['noEnclosedPorch']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noEnclosedPorch'][train['EnclosedPorch'] == 0]  = 1

index = train['EnclosedPorch'] > 0

sns.distplot(train['EnclosedPorch'][index], fit=norm);
fig = plt.figure()
res = stats.probplot(train['EnclosedPorch'][index], plot=plt)
fig = plt.figure()
plt.plot(train['EnclosedPorch'],train['SalePrice'],'o')


#test
test['noEnclosedPorch']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noEnclosedPorch'][test['EnclosedPorch'] == 0]  = 1
# 3SsnPorch
train = train.drop('3SsnPorch',axis=1)

#test
test = test.drop('3SsnPorch',axis=1)
# ScreenPorch
train['noScreenPorch']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noScreenPorch'][train['ScreenPorch'] == 0]  = 1

index = train['ScreenPorch'] > 0
train['ScreenPorch'] = np.log1p(train['ScreenPorch'])
sns.distplot(train['ScreenPorch'][index], fit=norm);
fig = plt.figure()
res = stats.probplot(train['ScreenPorch'][index], plot=plt)
fig = plt.figure()
plt.plot(train['ScreenPorch'][index],train['SalePrice'][index],'o')

#test
test['noScreenPorch']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noScreenPorch'][test['ScreenPorch'] == 0]  = 1
test['ScreenPorch'] = np.log1p(test['ScreenPorch'])
# PoolArea
train = train.drop('PoolArea',axis=1)
train = train.drop('PoolQC',axis=1)

#test
test = test.drop('PoolArea',axis=1)
test = test.drop('PoolQC',axis=1)
# Fence
# print(train['MiscVal'].value_counts())
# print(train['MiscVal'].isnull().sum()/len(train))
train = train.drop('Fence',axis=1)
train = train.drop('MiscFeature',axis=1)
train = train.drop('MiscVal',axis=1)
train = train.drop('MoSold',axis=1)
train = train.drop('YrSold',axis=1)

#test
test = test.drop('Fence',axis=1)
test = test.drop('MiscFeature',axis=1)
test = test.drop('MiscVal',axis=1)
test = test.drop('MoSold',axis=1)
test = test.drop('YrSold',axis=1)
# MoSold
# print(train['SaleType'].value_counts())
# print(train['SaleType'].isnull().sum()/len(train))

plt.plot(train['GrLivArea'][train['SaleType'] == 'WD'],train['SalePrice'][train['SaleType'] == 'WD'],'o')
plt.plot(train['GrLivArea'][train['SaleType'] == 'New'],train['SalePrice'][train['SaleType'] == 'New'],'o')
plt.plot(train['GrLivArea'][train['SaleType'] == 'COD'],train['SalePrice'][train['SaleType'] == 'COD'],'o')

#test
test['SaleType'] = test['SaleType'].fillna('WD')
# MasVnrType
# print(train['MasVnrType'].value_counts())
train['MasVnrType'] = train['MasVnrType'].fillna('None')
plt.plot(train['GrLivArea'][train['MasVnrType'] == 'None'],train['SalePrice'][train['MasVnrType'] == 'None'],'o')
plt.plot(train['GrLivArea'][train['MasVnrType'] == 'BrkFace'],train['SalePrice'][train['MasVnrType'] == 'BrkFace'],'o')
plt.plot(train['GrLivArea'][train['MasVnrType'] == 'Stone'],train['SalePrice'][train['MasVnrType'] == 'Stone'],'o')

#test
test['MasVnrType'] = test['MasVnrType'].fillna('None')
# MasVnrArea
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['noMasVnrArea']  = pd.Series(np.zeros((len(train))),index = train.index)
train['noMasVnrArea'][train['MasVnrArea'] == 0]  = 1

index = train['MasVnrArea'] > 0
train['MasVnrArea'] = np.log1p(train['MasVnrArea'])
sns.distplot(train['MasVnrArea'][index], fit=norm);
fig = plt.figure()
res = stats.probplot(train['MasVnrArea'][index], plot=plt)
fig = plt.figure()
plt.plot(train['MasVnrArea'][index],train['SalePrice'][index],'o')

#test
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
test['noMasVnrArea']  = pd.Series(np.zeros((len(test))),index = test.index)
test['noMasVnrArea'][test['MasVnrArea'] == 0]  = 1
test['MasVnrArea'] = np.log1p(test['MasVnrArea'])
# BsmtFinType2
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('Unf')
# print(train['BsmtFinType2'].value_counts())
# print(train['BsmtFinType2'].isnull().sum()/len(train))

#test
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('Unf')
# print(train['Exterior2nd'].value_counts())
# print(train['Exterior2nd'].isnull().sum()/len(train))

test['KitchenQual'] = test['KitchenQual'].fillna('TA')
test['Functional'] = test['Functional'].fillna('Typ')
test['Exterior1st'] = test['Exterior1st'].fillna('unknown')
test['Exterior2nd'] = test['Exterior2nd'].fillna('unknown')
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
print(train.isnull().sum().max())
# print(test.isnull().sum().sort_values(ascending=False))
print(test.isnull().sum().max())
# 相关性分析
corrmat = train.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cm = train[cols].corr()
f, ax = plt.subplots(figsize=(14, 11))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f')
plt.show()
concat_data = pd.concat([train,test])

# one-hot
dummies_data = pd.get_dummies(concat_data.drop('SalePrice', axis=1))

# 归一化
dummies_data = StandardScaler().fit_transform(dummies_data)
X = dummies_data[:len(train)]
print(X.shape)
submission_data = dummies_data[-len(test):]
print(submission_data.shape)
y = concat_data.iloc[:len(train)]['SalePrice'].values
# one-hot
# dummies_data = pd.get_dummies(train.drop('SalePrice', axis=1))
# X = StandardScaler().fit_transform(dummies_data)
# y = train['SalePrice'].values
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
alphas = [.0001, .0003, .0005, .0007, .0009, .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50,100,200,300,500]

scores = [np.sqrt(-cross_val_score(Ridge(alpha), X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()
          for alpha in alphas]

plt.plot(alphas, scores, label=Ridge.__name__)
plt.legend(loc='center')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

reg = linear_model.Ridge(200)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('均方根误差:',mean_squared_error(y_test,y_pred)**0.5)
import xgboost as xgb

regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

regr.fit(X_train,y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(X_test)
print('均方根误差:',mean_squared_error(y_test,y_pred)**0.5)

# Run prediction on the Kaggle test set.
y_pred_xgb = regr.predict(submission_data)
y_pred_xgb = np.expm1(y_pred_xgb)
# preds = []
# for i in range(10):
#     data_x,data_y = resample(X,y,n_samples=len(X))
#     reg = linear_model.Ridge(75)
#     reg.fit(data_x,data_y)
#     y_test_ridge = reg.predict(submission_data)
#     y_test_ridge = np.expm1(y_test_ridge)
#     preds.append(y_test_ridge)
# preds.append(y_pred_xgb)
# result = np.sum(preds,axis=0)/len(preds)

reg = linear_model.Ridge(200)
reg.fit(X,y)
y_test_ridge = reg.predict(submission_data)
y_test_ridge = np.expm1(y_test_ridge)
result = (y_test_ridge+y_pred_xgb)/2

plt.plot(np.expm1(test['GrLivArea']),result,'o')
my_submission = pd.DataFrame({'Id':test.index,'SalePrice': result})
my_submission.to_csv('submission.csv', index=False)
