import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.info()
train_df['PoolArea'].value_counts()
train_df['MiscVal'].value_counts()
to_drop = []

to_drop.extend(['Alley', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal' 'SaleType', 'SaleCondition'])
eng_feat_names = []

eng_feat_func = []
print(train_df.groupby('MoSold')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['MoSold'], y=train_df['SalePrice'], ax=ax)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.regplot(x=train_df['YearBuilt'], y=train_df['SalePrice'], ax=ax)

plt.show()
train_df['Sold_Age'] = train_df['YrSold'] - train_df['YearBuilt']

print(train_df['Sold_Age'].describe())
fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.regplot(x=train_df['Sold_Age'], y=train_df['SalePrice'], ax=ax)

plt.show()
to_drop.extend(['YearBuilt', 'YrSold'])
len(train_df[train_df['YearBuilt'] != train_df['YearRemodAdd']])
def was_remodeled(row):

    yr_blt = row['YearBuilt']

    yr_remodeled = row['YearRemodAdd']

    if yr_blt == yr_remodeled:

        return 0

    else:

        return 1
train_df['remodeled'] = train_df.apply(was_remodeled, axis=1)
print(train_df['remodeled'].value_counts())

print(train_df.groupby('remodeled')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['remodeled'], y=train_df['SalePrice'], ax=ax)

plt.show()
def yrs_remod(row):

    yr_blt = row['YearBuilt']

    yr_remodeled = row['YearRemodAdd']

    yr_sold = row['YrSold']

    if yr_blt == yr_remodeled:

        return 0

    else:

        return yr_sold - yr_remodeled
train_df['yrs_since_remod'] = train_df.apply(yrs_remod, axis=1)
fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x=train_df['yrs_since_remod'],y=train_df['SalePrice'],ax=ax)

plt.show()
to_drop.append('YearRemodAdd')
eng_feat_names.append('yrs_since_remod')

eng_feat_func.append(yrs_remod)
fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x=train_df['LotArea'],y=train_df['SalePrice'],ax=ax)

plt.show()
train_df[train_df['LotArea']>100000]
train_df['lot_to_bldg'] = train_df['1stFlrSF'] / train_df['LotArea']

fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x = train_df['lot_to_bldg'], y = train_df['SalePrice'], ax = ax)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x = train_df['LotFrontage'], y = train_df['SalePrice'], ax = ax)

plt.show()
train_df[train_df['LotFrontage']>300]
print(train_df['LotShape'].value_counts())

print(train_df.groupby('LotShape')['SalePrice'].median())
def lot_shape(row):

    if row['LotShape'] == 'Reg':

        return 'Reg'

    else:

        return 'Irreg'
train_df['LotShape'] = train_df.apply(lot_shape, axis=1)
print(train_df['LotShape'].value_counts())

print(train_df.groupby('LotShape')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['LotShape'], y=train_df['SalePrice'], ax=ax)

plt.show()
eng_feat_names.append('LotShape')

eng_feat_func.append(lot_shape)
print(train_df['LandContour'].value_counts())

print(train_df.groupby('LandContour')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['LandContour'], y=train_df['SalePrice'], ax=ax)

plt.show()
print(train_df['LotConfig'].value_counts())

print(train_df.groupby('LotConfig')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['LotConfig'], y=train_df['SalePrice'], ax=ax)

plt.show()
print(train_df['LandSlope'].value_counts())

print(train_df.groupby('LandSlope')['SalePrice'].median())
def land_slope(row):

    if row['LandSlope'] == 'Gtl':

        return 'Gtl'

    else:

        return 'Slope'
train_df['LandSlope'] = train_df.apply(land_slope, axis=1)
print(train_df['LandSlope'].value_counts())

print(train_df.groupby('LandSlope')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['LandSlope'], y=train_df['SalePrice'], ax=ax)

plt.show()
eng_feat_names.append('LandSlope')

eng_feat_func.append(land_slope)
fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x = train_df['GrLivArea'], y = train_df['SalePrice'], ax = ax)

plt.show()
train_df[train_df['GrLivArea'] > 4000]
train_df[['1stFlrSF', '2ndFlrSF', 'GrLivArea']].head(10)
len(train_df[(train_df['1stFlrSF'] + train_df['2ndFlrSF']) != train_df['GrLivArea']])
train_df[(train_df['1stFlrSF'] + train_df['2ndFlrSF']) != train_df['GrLivArea']].head()
to_drop.extend(['1stFlrSF', '2ndFlrSF'])
train_df[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].head(10)
len(train_df[(train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] +  train_df['BsmtUnfSF']) != train_df['TotalBsmtSF']])
fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x = train_df['BsmtUnfSF'], y = train_df['SalePrice'], ax = ax)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(20, 10))

sns.regplot(x = train_df['TotalBsmtSF'], y = train_df['SalePrice'], ax = ax)

plt.show()
train_df[train_df['TotalBsmtSF']>6000]
to_drop.extend(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'])
len(train_df[train_df['LowQualFinSF'] > 0])
to_drop.append('LowQualFinSF')
print(train_df['RoofMatl'].value_counts())

print(train_df.groupby('RoofMatl')['SalePrice'].median())
def rfmat(row):

    if row['RoofMatl'] == 'CompShg':

        return 'CompShg'

    elif row['RoofMatl'] == 'Tar&Grv':

        return 'Tar&Grv'

    elif row['RoofMatl'] == 'WdShake' or row['RoofMatl'] == 'WdShngl':

        return 'WdRoof'

    else:

        return 'OtherRoof'
train_df['RoofMatl'] = train_df.apply(rfmat, axis=1)
print(train_df['RoofMatl'].value_counts())

print(train_df.groupby('RoofMatl')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['RoofMatl'], y=train_df['SalePrice'], ax=ax)

plt.show()
eng_feat_names.append('RoofMatl')

eng_feat_func.append(rfmat)
fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['BsmtFullBath'], y = train_df['SalePrice'], ax = ax)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['BsmtHalfBath'], y = train_df['SalePrice'], ax = ax)

plt.show()
train_df['BsmtBaths'] = train_df['BsmtFullBath'] + .5*train_df['BsmtHalfBath']

fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['BsmtBaths'], y = train_df['SalePrice'], ax = ax)

plt.show()
to_drop.extend(['BsmtHalfBath', 'BsmtFullBath'])
fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['HalfBath'], y = train_df['SalePrice'], ax = ax)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['FullBath'], y = train_df['SalePrice'], ax = ax)

plt.show()
train_df['Baths'] = train_df['FullBath'] + .5*train_df['HalfBath']

fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['Baths'], y = train_df['SalePrice'], ax = ax)

plt.show()
to_drop.extend(['HalfBath', 'FullBath'])
len(train_df[train_df['TotRmsAbvGrd'] != train_df['BedroomAbvGr'] + train_df['KitchenAbvGr']])
train_df['OtherRmsAbvGr'] = train_df['TotRmsAbvGrd'] - train_df['BedroomAbvGr'] - train_df['KitchenAbvGr']
fig, ax = plt.subplots(1,1,figsize=(8, 5))

sns.regplot(x = train_df['OtherRmsAbvGr'], y = train_df['SalePrice'], ax = ax)

plt.show()
to_drop.append('TotRmsAbvGrd')
def newer_garage(row):

    garage_yr = row['GarageYrBlt']

    house_yr = row['YearBuilt']

    

    if garage_yr > house_yr:

        return 'yes'

    else:

        return 'no'
train_df['newer_garage'] = train_df.apply(newer_garage, axis=1)
print(train_df['newer_garage'].value_counts())

print(train_df.groupby('newer_garage')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['newer_garage'], y=train_df['SalePrice'], ax=ax)

plt.show()
to_drop.append('GarageYrBuilt')
eng_feat_names.append('newer_garage')

eng_feat_func.append(newer_garage)
print(train_df['OverallQual'].value_counts())

print(train_df.groupby('OverallQual')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['OverallQual'], y=train_df['SalePrice'], ax=ax)

plt.show()
print(train_df['OverallCond'].value_counts())

print(train_df.groupby('OverallCond')['SalePrice'].median())

fig, ax = plt.subplots(1,1,figsize=(8,5))

sns.boxplot(x=train_df['OverallCond'], y=train_df['SalePrice'], ax=ax)

plt.show()
to_drop.extend(['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'])