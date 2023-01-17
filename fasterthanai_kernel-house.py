# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/home-data-for-ml-course/train.csv'

path2 = '/kaggle/input/home-data-for-ml-course/test.csv'

train = pd.read_csv(path,index_col='Id')

test = pd.read_csv(path2,index_col='Id')

all = pd.concat([train,test])

all
with_str = {}

with_nan = {}

for each in all.columns:

    tp = all[each].dtype

    a = all[each].isnull().sum()

    if a > 0:

        with_nan[each] = a

    if tp == 'object' and each in list(with_nan.keys()):

        with_str[each] = 'with '+str(a)+'nan'

    if tp == 'object' and each not in list(with_nan.keys()):

        with_str[each] = 'complete'
print(all.dtypes)

sns.distplot(all.SalePrice,kde=False);
plt.figure(figsize=(16,10))

sns.heatmap(train[train.corr().iloc[-1][train.corr().iloc[-1]>0.5].index].corr(),annot=True,cmap="YlGnBu")
plt.figure(figsize=(12,12))

sns.catplot(x="OverallQual", y="SalePrice", kind="box", data=train);
#plt.figure(figsize=(26,15))

sns.regplot(x="GrLivArea", y="SalePrice", data=train);
to_be_check = train[(train.GrLivArea > 4500) & (train.SalePrice < 300000)]#.index.tolist()
to_be_check = to_be_check.append(train[(train.OverallQual == 4) & (train.SalePrice > 200000)])
all['PoolQC'][all.PoolQC.isnull()] = 'None'
dict1 = {'None' : 0, 'Po': 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5}

dict1['None']

all['PoolQC'] = all['PoolQC'].apply(lambda x:dict1[x])
all.groupby('PoolQC').PoolQC.count()
all[(all['PoolQC']==0) & (all['PoolArea'] > 0)].loc[:,['PoolArea', 'PoolQC', 'OverallQual']]
all.PoolQC[2421] = 2

all.PoolQC[2504] = 3

all.PoolQC[2600] = 2
all['MiscFeature'][all.MiscFeature.isnull()] = 'None'
sns.catplot(x="MiscFeature", y="SalePrice", kind="bar", data=all);
all['Alley'][all.Alley.isnull()]='None'
sns.catplot(x="Alley", y="SalePrice", kind="bar", data=all);
all['Fence'][all.Fence.isnull()]='None'
all.groupby('Fence').agg({'SalePrice':'median','Fence':'count'})#pd.DataFrame({'Fence_num':all[all.SalePrice.notnull()].groupby('Fence').Fence.count(),'SalePrice_median':all[all.SalePrice.notnull()].groupby('Fence').SalePrice.median()})   
all.FireplaceQu[all.FireplaceQu.isnull()]='None'
all['FireplaceQu'] = all['FireplaceQu'].apply(lambda x:dict1[x])
all.groupby('FireplaceQu').FireplaceQu.count()
plt.figure(figsize=(15,10))



sns.catplot(x="Neighborhood", y="LotFrontage", kind="bar", data=all).set_xticklabels(rotation=70)
medians = all.groupby('Neighborhood').LotFrontage.median()#.loc['Blmngtn']

for i in all[all.LotFrontage.isnull()].index:

    all.loc[i,'LotFrontage'] = medians.loc[all.iloc[i].Neighborhood]

    
all.LotShape = all.LotShape.apply(lambda x:{'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}[x])
all.groupby('LotShape').LotShape.count()
all.GarageYrBlt[all.GarageYrBlt.isnull()] = all.YearBuilt[all.GarageYrBlt.isnull()]
len(all[all.GarageType.isnull()&all.GarageQual.isnull()&all.GarageFinish.isnull()])
all[all.GarageType.notnull()&all.GarageQual.isnull()&all.GarageFinish.isnull()].index
all.loc[[2127,2577]][['GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish']]
all.groupby('GarageCond').GarageCond.count()
all.groupby('GarageQual').GarageCond.count()
all.groupby('GarageFinish').GarageCond.count()
all.loc[2127,'GarageCond'] = 'TA'

all.loc[2127,'GarageQual'] = 'TA'

all.loc[2127,'GarageFinish'] = 'Unf'
all.loc[[2127,2577]][['GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish']]
all.loc[2577,'GarageCars'] = 0

all.loc[2577,'GarageArea'] = 0

all.loc[2577,'GarageType'] = 'No Garage'
all.loc[[2127,2577]][['GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish']]
all.GarageType[all.GarageType.isnull()]='No Garage'
all.GarageFinish[all.GarageFinish.isnull()]='None'
all.GarageFinish = all.GarageFinish.apply(lambda x:{'None':0, 'Unf':1, 'RFn':2, 'Fin':3}[x])
all.groupby('GarageFinish').GarageFinish.count()
all.GarageQual[all.GarageQual.isnull()]='None'
all.GarageQual = all.GarageQual.apply(lambda x: dict1[x])
all.groupby('GarageQual').GarageQual.count()
all.GarageCond[all.GarageCond.isnull()]='None'
all.GarageCond = all.GarageCond.apply(lambda x: dict1[x])
all.groupby('GarageCond').GarageCond.count()
with_nan
all[all.BsmtFinType1.notnull() & (all.BsmtQual.isnull() | all.BsmtCond.isnull() | all.BsmtExposure.isnull() | all.BsmtFinType2.isnull())][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
all[all.BsmtFinType1.isnull()].BsmtFullBath.unique()
all.groupby('BsmtFinType2').BsmtFinType2.count()
all.groupby('BsmtExposure').BsmtExposure.count()
all.groupby('BsmtCond').BsmtCond.count()
all.groupby('BsmtQual').BsmtQual.count()
all.loc[333,'BsmtFinType2' ] = 'Unf'

all.loc[[949,1488,2349],'BsmtExposure' ] = 'No'

all.loc[[2186,2041,2525],'BsmtCond' ] = 'TA'

all.loc[[2218,2219],'BsmtQual' ] = 'TA'
all[all.BsmtFinType1.notnull() & (all.BsmtQual.isnull() | all.BsmtCond.isnull() | all.BsmtExposure.isnull() | all.BsmtFinType2.isnull())][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
all.BsmtQual[all.BsmtQual.isnull()] = 'None'

all.BsmtExposure[all.BsmtExposure.isnull()] = 'None'

all.BsmtCond[all.BsmtCond.isnull()] = 'None'

all.BsmtFinType1[all.BsmtFinType1.isnull()] = 'None'

all.BsmtFinType2[all.BsmtFinType2.isnull()] = 'None'
all.BsmtQual = all.BsmtQual.apply(lambda x:dict1[x])

all.BsmtCond = all.BsmtCond.apply(lambda x:dict1[x])

all.BsmtExposure = all.BsmtExposure.apply(lambda x:{'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}[x])

all.BsmtFinType1 = all.BsmtFinType1.apply(lambda x:{'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}[x])

all.BsmtFinType2 = all.BsmtFinType2.apply(lambda x:{'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}[x])


all[all.BsmtQual.isnull() | (all.BsmtFullBath.isnull() | all.BsmtHalfBath.isnull() | all.BsmtFinSF1.isnull() | all.BsmtFinSF2.isnull() | all.BsmtUnfSF.isnull() |all.TotalBsmtSF.isnull() )][['BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
all.loc[[2189,2121],['BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]=0
all.loc[[2121,2189]][['BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
len(all[all.MasVnrType.isnull()&all.MasVnrArea.isnull()])
all[all.MasVnrType.isnull() & all.MasVnrArea.notnull()][['MasVnrType' , 'MasVnrArea']]
all.groupby('MasVnrType').MasVnrType.count()
all.loc[2611,'MasVnrType']='BrkFace'
all[all.MasVnrType.isnull() & all.MasVnrArea.notnull()][['MasVnrType' , 'MasVnrArea']]
all.MasVnrType[all.MasVnrType.isnull()] = 'None'
all.MasVnrArea[all.MasVnrArea.isnull()] = 0

all.groupby('MasVnrArea').MasVnrArea.count()
def F_T(all,name,sth='None',dict1=None,mode=False):

    if mode==False:

        all[name][all[name].isnull()]=sth

        print(all.groupby(name)[name].count())

    if mode == True:

        all[name][all[name].isnull()]=all.groupby(name)[name].count().idxmax()

    if dict1!=None:

        all[name] = all[name].apply(lambda x:dict1[x])

    return all.groupby(name)[name].count()
F_T(all,'MasVnrType',dict1={'None':0, 'BrkCmn':0, 'BrkFace':1, 'Stone':2})
all.groupby('MasVnrType')['MasVnrType'].count()
F_T(all,'MSZoning',mode=True)
all[all['KitchenQual'].isnull()][['KitchenQual','KitchenAbvGr']]
all[all['KitchenAbvGr']==0][['KitchenQual','KitchenAbvGr']]
def assign(all,dict1={},value=[]):

    #dict1={name:[id,id,id]}

    #value=[a,b,c]

    for i,each in enumerate(dict1.items()):

        all.loc[each[1],[each[0]]] = value[i]
assign(all,{'KitchenAbvGr':[955,2588,2860]},[1])

F_T(all,'KitchenQual',mode=True,dict1=dict1)

all.Utilities = None

F_T(all,dict1={'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7},name='Functional',mode=True)

F_T(all,name='Exterior1st',mode=True)

F_T(all,name='Exterior2nd',mode=True)

F_T(all,name='ExterQual',dict1=dict1)

F_T(all,name='ExterCond',dict1=dict1)

F_T(all,name='Electrical',mode=True)

F_T(all,name='SaleType',mode=True)
F_T(all,name='HeatingQC',dict1=dict1)

F_T(all,name='CentralAir',dict1={'N':0, 'Y':1})

F_T(all,name='LandSlope',dict1={'Sev':0, 'Mod':1, 'Gtl':2})

F_T(all,name='Street',dict1={'Grvl':0, 'Pave':1})

F_T(all,name='PavedDrive',dict1={'N':0, 'P':1, 'Y':2})
all.YrSold = all.YrSold.apply(lambda x:str(x))

all.MoSold = all.MoSold.apply(lambda x:str(x))
sns.catplot(x='YrSold', y="SalePrice", kind="bar", data=all);
sns.catplot(x='MoSold', y="SalePrice", kind="bar", data=all);
dict2={'20':'1 story 1946+', '30':'1 story 1945-', '40':'1 story unf attic', '45':'1,5 story unf', '50':'1,5 story fin', '60':'2 story 1946+', '70':'2 story 1945-', '75':'2,5 story all ages', '80':'split/multi level', '85':'split foyer', '90':'duplex all style/age', '120':'1 story PUD 1946+', '150':'1,5 story PUD all', '160':'2 story PUD 1946+', '180':'PUD multilevel', '190':'2 family conversion'}
all.MSSubClass = all.MSSubClass.apply(lambda x:str(x))

F_T(all,name='MSSubClass',dict1=dict2)
plt.figure(figsize=(16,10))

sns.heatmap(train[train.corr().iloc[-1][train.corr().iloc[-1]>0.5].index].corr(),annot=True,cmap="YlGnBu")
# from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(n_estimators=100, criterion='mse')

# model.fit(train.iloc[:,:-1],train.SalePrice)

# pre = model.predict
train.iloc[:,:-1]
plt.figure(figsize=(20,10))

plt.subplot(2,4,1)

sns.kdeplot(all.GrLivArea)

plt.subplot(2,4,2)

sns.distplot(all.TotRmsAbvGrd,kde=False)

plt.subplot(2,4,3)

sns.kdeplot(all['1stFlrSF'])

plt.subplot(2,4,4)

sns.kdeplot(all['2ndFlrSF'])

plt.subplot(2,4,5)

sns.kdeplot(all.TotalBsmtSF)

plt.subplot(2,4,6)

sns.distplot(all.LotArea,kde=False)

plt.subplot(2,4,7)

sns.kdeplot(all.LotFrontage)

plt.subplot(2,4,8)

sns.distplot(all.LowQualFinSF,kde=False)

all.loc[:,['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']]#后三个加和等于第一个
plt.figure(figsize=(20,10))

sns.catplot(x="Neighborhood", kind="count", palette="ch:.25", data=all).set_xticklabels(rotation=60)
plt.subplot(3,3,1)

sns.distplot(all.OverallQual,kde=False)

plt.subplot(3,3,2)

sns.distplot(all.ExterQual,kde=False)

plt.subplot(3,3,3)

sns.distplot(all.BsmtQual,kde=False)

plt.subplot(3,3,4)

sns.distplot(all.KitchenQual,kde=False)

plt.subplot(3,3,5)

sns.distplot(all.GarageQual,kde=False)

plt.subplot(3,3,6)

sns.distplot(all.FireplaceQu,kde=False)

plt.subplot(3,3,7)

sns.catplot(x="PoolQC", kind="count", palette="ch:.25", data=all)

plt.show()
sns.catplot(x="MSSubClass", kind="count", palette="ch:.25", data=all).set_xticklabels(rotation=60)
sns.catplot(x="MSSubClass", y="SalePrice", kind="bar", data=all).set_xticklabels(rotation=60)
# from sklearn.ensamble import RandomForest

# from sklearn import shffle

# model1 = RandomForest(n_estmator=100,metrics='mse')

# model2=RandomForest(n_estmator=100,metrics='mse')

# for each in all.columns:

#     data1=train.copy()

#     data2=test.copy()

#     y=data1.SalePrice

#     x=data1.drop(['SalePrice'],axis=1)

#     a=data2

#     model1.fit(x,y)

#     shuffle(x[each])

#     model2.fit(x,y)

#     model1.predict


all['TotBathrooms'] =  all.FullBath+ all.HalfBath*0.5 +all.BsmtFullBath+ all.BsmtHalfBath*0.5
all
sns.regplot(x=all.TotBathrooms,y=all.SalePrice)
sns.catplot(x="TotBathrooms", kind="count", palette="ch:.25", data=all)
all.YearRemodAdd.unique()
all.loc[:,['TotBathrooms','SalePrice']].corr()
all['Remod'] = (all.YearBuilt==all.YearRemodAdd).apply(lambda x:int(x))
all['Age'] = all.YrSold.copy().apply(lambda x:int(x))-all.YearRemodAdd
plt.figure(figsize=(20,10))

sns.regplot(x=all.Age,y=all.SalePrice)
all.loc[:,['Age','SalePrice']].corr()
sns.catplot(x="Remod", y="SalePrice", kind="bar", data=all)
all['IsNew'] = (all.YrSold.copy().apply(lambda x:int(x))==all.YearBuilt).apply(lambda x:int(x))
all.YearBuilt.dtype
sns.catplot(x="IsNew", y="SalePrice", kind="bar", data=all)
order=list(all.groupby('Neighborhood').SalePrice.median().sort_values().index)

sns.catplot(x="Neighborhood", y="SalePrice", kind="bar", data=all,order=order).set_xticklabels(rotation=70)
order
all['NeighRich']=0

all['NeighRich'][all.loc[:,'Neighborhood'].apply(lambda x:x in ['StoneBr', 'NridgHt', 'NoRidge'])]=4

all['NeighRich'][all.loc[:,'Neighborhood'].apply(lambda x:x in ['OldTown', 'Edwards', 'BrkSide', 'Sawyer', 'Blueste', 'SWISU', 'NAmes', 'NPkVill', 'Mitchel'])]=1

all['NeighRich'][all.loc[:,'Neighborhood'].apply(lambda x:x in ['SawyerW', 'Gilbert', 'NWAmes', 'Blmngtn', 'CollgCr', 'ClearCr', 'Crawfor'])]=2

all['NeighRich'][all.loc[:,'Neighborhood'].apply(lambda x:x in ['Veenker', 'Somerst', 'Timber'])]=3

all['NeighRich'][all.loc[:,'Neighborhood'].apply(lambda x:x in ['MeadowV', 'IDOTRR', 'BrDale'])] = 0
all['NeighRich']
plt.figure(figsize=(20,10))

all['TotalSqFeet'] = all.GrLivArea + all.TotalBsmtSF

sns.regplot(x='TotalSqFeet',y = 'SalePrice',data=all)
all.loc[:,['TotalSqFeet','SalePrice']].corr()
all= all.drop(['YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'BsmtFinSF1'],axis=1)
all=all.drop([524,1299],axis=0)
all=all.drop(['Utilities'],axis=1)