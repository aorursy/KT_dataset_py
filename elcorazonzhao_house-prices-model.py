import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import math
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.info()
train_corr = train.corr()

train_corr = train_corr.applymap(lambda x: -1 if x<-0.4 else x)

train_corr = train_corr.applymap(lambda x: 0 if -0.4<=x<=0.4 else x)

train_corr = train_corr.applymap(lambda x: 1 if x>0.4 else x)

sns.heatmap(train_corr,vmin=-1,vmax=1,cmap=sns.color_palette('RdBu',n_colors=8))

plt.show()
train['MSSubClass'].value_counts()

train.groupby(['MSSubClass'])['SalePrice'].mean().plot()
train['MSZoning'].value_counts()

train.groupby(['MSZoning'])['SalePrice'].mean().plot()
lotfrontage = train[train['LotFrontage'].notnull()]

lotfrontage['LotFrontage'].value_counts().sort_index().plot()
train['LotFrontage_group'] = pd.cut(lotfrontage['LotFrontage'],bins=[0,50,100,400],labels=['close','middle','far'])

train.groupby(['LotFrontage_group'])['SalePrice'].mean().plot()
train['LotArea'].value_counts().sort_index().plot()
train['LotArea_group'] = pd.cut(train['LotArea'],bins=[0,5000,10000,15000,20000,25000,2500000],labels=[1,2,3,4,5,6])

train.groupby(['LotArea_group'])['SalePrice'].mean().plot()
train['Street'].value_counts()
train.groupby(['Street'])['SalePrice'].mean()
train['Alley'].value_counts()

train.groupby(['Alley'])['SalePrice'].mean()

train['Alley_nan'] =  train['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1)

train.groupby(['Alley_nan'])['SalePrice'].mean().plot()
train['LotShape'].value_counts()

train.groupby(['LotShape'])['SalePrice'].mean().plot()
train['LandContour'].value_counts()

train.groupby(['LandContour'])['SalePrice'].mean().plot()
train['Utilities'].value_counts()

train.groupby(['Utilities'])['SalePrice'].mean()
train['LotConfig'].value_counts()

train.groupby(['LotConfig'])['SalePrice'].mean().plot()
train['LandSlope'].value_counts()

train.groupby(['LandSlope'])['SalePrice'].mean().plot()
train['Neighborhood'].value_counts()

neighborhood_price=train.groupby(['Neighborhood'])['SalePrice'].mean().sort_values()

sns.barplot(list(range(len(neighborhood_price))),neighborhood_price.values)

plt.xticks(list(range(len(neighborhood_price))),neighborhood_price.index)
#error

#train['Condition2'].value_counts()

#cdt1 = train.groupby(['Condition1'])['SalePrice'].mean().plot(label='cdt1')

#cdt2 = train.groupby(['Condition2'])['SalePrice'].mean().plot(label='cdt2')

#plt.xlabel('Condition')

#plt.legend()
train.groupby(['Condition1','Condition2'])['SalePrice'].mean().sort_values()
train['BldgType'].value_counts()

train.groupby(['BldgType'])['SalePrice'].mean().plot()
train['HouseStyle'].value_counts()

train.groupby(['HouseStyle'])['SalePrice'].mean().plot()
train['OverallQual'].value_counts()

train.groupby(['OverallQual'])['SalePrice'].mean().sort_index().plot()
train['OverallCond'].value_counts()

train.groupby(['OverallCond'])['SalePrice'].mean().sort_index().plot()
train['YearBuilt'].value_counts()

train.groupby(['YearBuilt'])['SalePrice'].mean().sort_index().plot()
train['YearRemodAdd'].value_counts()

train.groupby(['YearRemodAdd'])['SalePrice'].mean().sort_index().plot()
train['YearRemod_yr'] = train['YearRemodAdd']-train['YearBuilt']

train['YearRemod_yn'] = train['YearRemod_yr'].apply(lambda x: 1 if x !=0 else 0)

#train['YearRemod_yn'].value_counts()

train.groupby(['YearRemod_yn'])['SalePrice'].mean()

#train['YearRemod_yr'].value_counts()

#train.groupby(['YearRemod_yr'])['SalePrice'].mean().sort_index().plot()

#train['YearRemod_yr'] = pd.cut(train['YearRemod_yr'],bins=[0,1,10,20,50,100,200],labels=['0-1','1-10','10-20','20-50','50-100','100+'])

#train.groupby(['YearRemod_yr'])['SalePrice'].mean().plot()
train['RoofStyle'].value_counts()

train.groupby(['RoofStyle'])['SalePrice'].mean().plot()
train['RoofMatl'].value_counts()

train.groupby(['RoofMatl'])['SalePrice'].mean().plot()
#train['Exterior1st'].value_counts()

#ex1 = train.groupby(['Exterior1st'])['SalePrice'].mean()



#train['Exterior2nd'].value_counts()

#ex2 = train.groupby(['Exterior2nd'])['SalePrice'].mean()



#sns.barplot(list(range(len(ex1))),ex1.values,color='R',alpha=0.5,label='ex1')

#sns.barplot(list(range(len(ex2))),ex2.values,color='B',alpha=0.5,label='ex2')

#plt.xticks(list(range(len(ex1))),ex1.index)

#plt.legend()
train.groupby(['Exterior1st','Exterior2nd'])['SalePrice'].mean().sort_values()
train['MasVnrType'].value_counts()

train.groupby(['MasVnrType'])['SalePrice'].mean().sort_values().plot()
train['MasVnrArea'].value_counts().sort_index()

MVA_not_miss = train[train['MasVnrArea'].notnull()]

MVA_not_miss['MasVnrArea'] = pd.cut(MVA_not_miss['MasVnrArea'],20,labels=list(range(20)))

MVA_not_miss.groupby(['MasVnrArea'])['SalePrice'].mean()
train['ExterQual'].value_counts()

train.groupby(['ExterQual'])['SalePrice'].mean().sort_values().plot()
train['ExterCond'].value_counts()

train.groupby(['ExterCond'])['SalePrice'].mean().sort_values().plot()
train.groupby(['ExterQual','ExterCond'])['SalePrice'].mean().sort_values()
train['Foundation'].value_counts()

train.groupby(['Foundation'])['SalePrice'].mean().sort_values().plot()
train['BsmtQual'].value_counts(dropna=False)

train.groupby(['BsmtQual'])['SalePrice'].mean().sort_values().plot()
train['BsmtCond'].value_counts(dropna=False)

train.groupby(['BsmtCond'])['SalePrice'].mean().sort_values().plot()
train['BsmtExposure'].value_counts(dropna=False)

train.groupby(['BsmtExposure'])['SalePrice'].mean().sort_values().plot()
train['BsmtFinType1'].value_counts(dropna=False)

train.groupby(['BsmtFinType1'])['SalePrice'].mean().sort_values().plot()
train['BsmtFinType2'].value_counts(dropna=False)

train.groupby(['BsmtFinType2'])['SalePrice'].mean().sort_values().plot()
train['BsmtFinSF1'].value_counts(dropna=False).sort_index()

bstmfinsf1_not_miss = train[train['BsmtFinSF1'].notnull()]

MVA_not_miss['BsmtFinSF1'] = pd.cut(bstmfinsf1_not_miss['BsmtFinSF1'],bins=[0,500,1000,1500,2000,2500,6000],labels=list(range(6)))

MVA_not_miss.groupby(['BsmtFinSF1'])['SalePrice'].mean().plot()
train['BsmtFinSF2'].value_counts(dropna=False).sort_index()

bstmfinsf2_not_miss = train[train['BsmtFinSF2'].notnull()]

MVA_not_miss['BsmtFinSF2'] = pd.cut(bstmfinsf2_not_miss['BsmtFinSF2'],bins=[0,500,1000,1500],labels=list(range(3)))

MVA_not_miss.groupby(['BsmtFinSF2'])['SalePrice'].mean().plot()
train.groupby(['BsmtFinType1','BsmtFinType2'])['SalePrice'].mean().sort_values().plot()
train['BsmtUnfSF'].value_counts(dropna=False).sort_index()

bstmunfsf_not_miss = train[train['BsmtUnfSF'].notnull()]

bstmunfsf_not_miss['BsmtUnfSF'] = pd.cut(bstmunfsf_not_miss['BsmtUnfSF'],6,labels=list(range(6)))

bstmunfsf_not_miss.groupby(['BsmtUnfSF'])['SalePrice'].mean().plot()
train['TotalBsmtSF'].value_counts(dropna=False).sort_index()

totalbsmtsf_not_miss = train[train['TotalBsmtSF'].notnull()]

totalbsmtsf_not_miss['TotalBsmtSF'] = pd.cut(totalbsmtsf_not_miss['TotalBsmtSF'],6,labels=list(range(6)))

totalbsmtsf_not_miss.groupby(['TotalBsmtSF'])['SalePrice'].mean()
train['Heating'].value_counts(dropna=False)

train.groupby(['Heating'])['SalePrice'].mean().sort_values().plot()
train['HeatingQC'].value_counts(dropna=False)

train.groupby(['HeatingQC'])['SalePrice'].mean().sort_values().plot()
train['CentralAir'].value_counts(dropna=False)
f = plt.figure()

f.add_subplot(111)

graph = sns.barplot(x='CentralAir',y='SalePrice',data=train)

for p in graph.patches:

    height = p.get_height()

    if math.isnan(height):

        height = 0

    graph.text(p.get_x()+p.get_width()/2.,height+5000,int(height),ha='center')
train['Electrical'].value_counts(dropna=False)

train.groupby(['Electrical'])['SalePrice'].mean().sort_values().plot()
train['1stFlrSF'].value_counts(dropna=False).sort_index()

fstflrsf_not_miss = train[train['1stFlrSF'].notnull()]

fstflrsf_not_miss['1stFlrSF'] = pd.cut(fstflrsf_not_miss['1stFlrSF'],5,labels=list(range(5)))

fstflrsf_not_miss.groupby(['1stFlrSF'])['SalePrice'].mean().plot()
train['2ndFlrSF'].value_counts(dropna=False).sort_index()

secflrsf_not_miss = train[train['2ndFlrSF'].notnull()]

secflrsf_not_miss['2ndFlrSF'] = pd.cut(secflrsf_not_miss['2ndFlrSF'],5,labels=list(range(5)))

secflrsf_not_miss.groupby(['2ndFlrSF'])['SalePrice'].mean().plot()
flr_diff = train['1stFlrSF'] - train['2ndFlrSF']

flr_diff.name = 'flr_diff'

train = train.merge(flr_diff,how='inner',on=train.index)
flrdiff_not_miss = train[train['flr_diff'].notnull()]

flrdiff_not_miss['flr_diff'] = pd.cut(flrdiff_not_miss['flr_diff'],bins=[-1000,0,1000,2000,3000],labels = [(-1000,0),(0,1000),(1000,2000),(2000,3000)])

flrdiff_not_miss.groupby(['flr_diff'])['SalePrice'].mean().plot()
train['LowQualFinSF'].value_counts(dropna=False).sort_index()

lowqfinsf_not_miss = train[train['LowQualFinSF'].notnull()]

lowqfinsf_not_miss['LowQualFinSF'] = pd.cut(lowqfinsf_not_miss['LowQualFinSF'],6,labels=list(range(6)))

lowqfinsf_not_miss.groupby(['LowQualFinSF'])['SalePrice'].mean().plot()
train['GrLivArea'].value_counts(dropna=False).sort_index()

grlivarea_not_miss = train[train['GrLivArea'].notnull()]

grlivarea_not_miss['GrLivArea'] = pd.cut(grlivarea_not_miss['GrLivArea'],6,labels=list(range(6)))

grlivarea_not_miss.groupby(['GrLivArea'])['SalePrice'].mean().plot()
bath_lst = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']

fig = []

for i in range(len(bath_lst)):

    print(train[bath_lst[i]].value_counts(dropna=False).sort_index())

    print('\n------------------')

    fig.append(i)

    fig[i] = plt.figure()

    fig[i].add_subplot(111)

    index = train.groupby(bath_lst[i])['SalePrice'].mean().index

    values = train.groupby(bath_lst[i])['SalePrice'].mean().values

    plt.plot(index,values)

    plt.title(bath_lst[i])
train['BedroomAbvGr'].value_counts(dropna=False).sort_index()

train.groupby(['BedroomAbvGr'])['SalePrice'].mean().plot()
train['KitchenAbvGr'].value_counts(dropna=False).sort_index()

train.groupby(['KitchenAbvGr'])['SalePrice'].mean().plot()
train['KitchenQual'].value_counts(dropna=False)

train.groupby(['KitchenQual'])['SalePrice'].mean().sort_values().plot()
train['TotRmsAbvGrd'].value_counts(dropna=False).sort_index()

train.groupby(['TotRmsAbvGrd'])['SalePrice'].mean().plot()
train['Functional'].value_counts(dropna=False)

train.groupby(['Functional'])['SalePrice'].mean().sort_values().plot()
train['Fireplaces'].value_counts(dropna=False).sort_index()

train.groupby(['Fireplaces'])['SalePrice'].mean().plot()
train['FireplaceQu'].value_counts(dropna=False)

train.groupby(['FireplaceQu'])['SalePrice'].mean().sort_values().plot()
train['GarageType'].value_counts(dropna=False)

train.groupby(['GarageType'])['SalePrice'].mean().sort_values().plot()
train['YearBuilt'].value_counts()

train.groupby(['YearBuilt'])['SalePrice'].mean().sort_index().plot(label='YearBuilt')

train['YearRemodAdd'].value_counts()

train.groupby(['YearRemodAdd'])['SalePrice'].mean().sort_index().plot(label='YearRemodAdd')

train['GarageYrBlt'].value_counts(dropna=False).sort_index()

train.groupby(['GarageYrBlt'])['SalePrice'].mean().sort_index().plot(label='GarageYrBlt')

plt.legend()
train['GarageFinish'].value_counts(dropna=False)

train.groupby(['GarageFinish'])['SalePrice'].mean().sort_values().plot()
train['GarageCars'].value_counts(dropna=False).sort_index()

train.groupby(['GarageCars'])['SalePrice'].mean().plot()
area_lst = ['GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']

fig = []

for i in range(len(area_lst)):

    print(train[area_lst[i]].value_counts(dropna=False).sort_index())

    print('------------------')

    fig.append(i)

    fig[i] = plt.figure()

    fig[i].add_subplot(111)

    area_not_miss = train[train[area_lst[i]].notnull()]

    area_not_miss[area_lst[i]] = pd.cut(area_not_miss[area_lst[i]],bins=6,labels=list(range(6)))

    area_not_miss.groupby(area_lst[i])['SalePrice'].mean().plot()

    plt.title(area_lst[i])
train['GarageQual'].value_counts(dropna=False)

train.groupby(['GarageQual'])['SalePrice'].mean().sort_values().plot()
train['GarageCond'].value_counts(dropna=False)

train.groupby(['GarageCond'])['SalePrice'].mean().sort_values().plot()
train.groupby(['GarageQual','GarageCond'])['SalePrice'].mean().sort_values().plot()
train['PavedDrive'].value_counts(dropna=False)

graph = sns.barplot(x='PavedDrive',y='SalePrice',data=train)

for p in graph.patches:

    height = p.get_height()

    if math.isnan(height):

        height = 0

    graph.text(p.get_x()+p.get_width()/2,height+5000,int(height),ha='center')
train['PoolArea'].value_counts(dropna=False).sort_index()

train.groupby(['PoolArea'])['SalePrice'].mean().sort_index().plot()
train['PoolQC'].value_counts(dropna=False)

train.groupby(['PoolQC'])['SalePrice'].mean().sort_values().plot()
train['Fence'].value_counts(dropna=False)

train.groupby(['Fence'])['SalePrice'].mean().sort_values().plot()
train['MiscFeature'].value_counts(dropna=False)

train.groupby(['MiscFeature'])['SalePrice'].mean().sort_values()
train['MiscVal'].value_counts(dropna=False)

#train.groupby(['MiscVal'])['SalePrice'].mean().plot()
train['MoSold'].value_counts(dropna=False)
train['YrSold'].value_counts(dropna=False)
train.groupby(['YrSold','MoSold'])['SalePrice'].mean().plot()
train['SaleType'].value_counts(dropna=False)

graph = sns.barplot(x='SaleType',y='SalePrice',data=train)

for p in graph.patches:

    height = p.get_height()

    if math.isnan(height):

        height = 0

    graph.text(p.get_x()+p.get_width()/2,height+5000,int(height),ha='center')
train['SaleCondition'].value_counts(dropna=False)

train.groupby(['SaleCondition'])['SalePrice'].mean().sort_values().plot()