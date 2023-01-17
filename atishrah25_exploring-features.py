import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv("train.csv")

test = pd.read_csv('test.csv')

Hid = test['Id']
print(train.shape)

print(test.shape)

print(train.index)

print(test.index)

train_y = train['SalePrice']

train_X = train.drop(['SalePrice'],axis=1)

print(train_X.columns)

ulimit = np.percentile(train_y.values,95)

llimit = np.percentile(train_y.values,2)

train_y.ix[train_y>ulimit]=ulimit

train_y.ix[train_y<llimit]=llimit
frames = [train_X,test]

combined = pd.concat(frames,ignore_index=True)

col = combined.columns



combined.isnull().sum()
for i in col:

    combined[i] = combined[i].fillna('none')
print(combined.head(9))

print(combined.tail(9))

print(train_X.head(9))

print(test.tail(9))
train.describe()

def binning(col,cut_pts,labels=None

            ):

    minval=col.min()

    maxval=col.max()

    break_pts = [minval]+cut_pts+[maxval]

    if not labels:

        labels = range(len(cut_pts)+1)

    col_bins=pd.cut(col,bins=break_pts,labels=labels,include_lowest=True)

    return col_bins



test.describe()
np.corrcoef(train_X['YearBuilt'],train_y)
train_X['MSSubClass'].unique()

col = 'MSSubClass'

unq = list(train_X[col].unique())

n = len(train_X[col].unique())

plt.figure(figsize=(25,9))

plt.subplot(121)

plt.scatter(train_X[col],train_y,marker='x')

plt.xlabel(col)

plt.ylabel('SalePrice')

plt.subplot(122)

sns.violinplot(train_X[col],train_y)    

plt.show()
lmean = []

u = train['MSSubClass'].unique()

for i in u:

    x = (train_y.loc[train['MSSubClass']==i]).mean()

    lmean.append(x)

plt.figure(figsize=(12,5))

plt.plot(u,lmean,marker='X')

plt.xticks(u,fontsize=8)

plt.xlabel('MSSubClass')

plt.ylabel('mean_sale_price')

plt.show()

df = pd.get_dummies(combined['MSSubClass'])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['MSSubClass'],axis=1)
col = 'MSZoning'

lmean = []

u = train['MSZoning'].unique()

for i in u:

    x = (train_y.loc[train['MSZoning']==i]).mean()

    lmean.append(x)

plt.figure(figsize=(20,10))

plt.subplot(131)

plt.plot(range(len(u)),lmean,marker='X')

plt.xticks(fontsize=8)

plt.xticks(range(len(u)),u)

plt.xlabel('MSZoning')

plt.ylabel('mean_sale_price')

plt.subplot(132)

train['MSZoning'].value_counts().plot.bar()

plt.ylabel('frequency')

plt.subplot(133)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

print(combined['RL'].loc[combined['RL']==1].shape)

ind = combined['FV'].loc[combined['FV']==1].index



combined['RL'].loc[ind] = 1

print(combined['RL'].loc[combined['RL']==1].shape)

ind = combined['RH'].loc[combined['RH']==1].index

combined['RM'].loc[ind] = 1

combined = combined.drop(['FV','C (all)','RH','MSZoning'],axis=1)



combined['LotFrontage'] = combined['LotFrontage'].replace(['none'],[52])

lotwidth = train_X['LotArea']/train_X['LotFrontage']

plt.figure(figsize=(18,5))

plt.subplot(131)

plt.scatter(train_X['LotFrontage'],train_y,marker='x',c='darkorange',alpha=0.8)

plt.xlabel('LotFrontage')

plt.ylabel('sale_price')

plt.subplot(132)

plt.scatter(train_X['LotArea'],train_y,marker='x',c='lightblue',alpha=1)

plt.xlabel('LotArea')

plt.subplot(133)

plt.scatter(lotwidth,train_y,marker='x',c='blue',alpha=.8)

plt.xlabel('LotWidth')

plt.show()
col = 'LotShape'

plt.figure(figsize=(18,9))

plt.subplot(121)

train[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()

df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

ind = combined['IR2'].loc[combined['IR2']==1].index

print(combined['IR1'].loc[combined['IR1']==1].shape)

combined['IR1'].loc[ind] = 1

print(combined['IR1'].loc[combined['IR1']==1].shape)

combined = combined.drop(['IR2','IR3','LotShape'],axis=1)

col = 'LotConfig'

plt.figure(figsize=(20,10))

plt.subplot(121)

train[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()



df = pd.get_dummies(combined[col])

combined= pd.concat([combined,df],axis=1)

combined = combined.drop(['FR2','FR3','LotConfig'],axis=1)

col ='LandContour'

plt.figure(figsize=(20,9))

plt.subplot(121)

train[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()

df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

ind = combined['Bnk'].loc[combined['Bnk']==1].index

print(combined['Lvl'].loc[combined['Lvl']==1].shape)

combined['Lvl'].loc[ind] = 1

print(combined['Lvl'].loc[combined['Lvl']==1].shape)

ind = combined['HLS'].loc[combined['HLS']==1].index

print(combined['Low'].loc[combined['Low']==1].shape)

combined['Low'].loc[ind] = 1

print(combined['Low'].loc[combined['Low']==1].shape)

combined= combined.drop(['Bnk','HLS','LandContour'],axis=1)

print(ind)

col1 = 'Alley'

col2 = 'Street'

train_X['Alley'] = train_X['Alley'].fillna('none')

train['Alley'] = train['Alley'].fillna('none')



plt.figure(figsize=(20,10))

plt.subplot(221)

train_X[col1].value_counts().plot.bar()

plt.xlabel('Alley')

plt.subplot(222)

sns.violinplot(train_X[col1],train_y)

plt.subplot(223)

train_X[col2].value_counts().plot.bar()

plt.xlabel('Street')

plt.subplot(224)

sns.violinplot(train_X[col2],train_y)

plt.show()
combined = combined.drop(['Street'],axis=1)
df = pd.get_dummies(combined[col1])

combined = pd.concat([combined,df],axis=1)



combined = combined.drop(['Alley'],axis=1)



combined = combined.drop(['Utilities'],axis=1)

train = train.drop(['Utilities'],axis=1)

col='LandSlope'

plt.figure(figsize=(20,9))

plt.subplot(121)

train[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()

df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

print(combined['Mod'].loc[combined['Mod']==1].shape)

inx = list(combined.loc[combined['Sev']==1].index)

combined.loc[inx,'Mod'] = 1

print(combined['Mod'].loc[combined['Mod']==1].shape)

print(inx)

combined = combined.drop(['LandSlope','Sev'],axis=1)

col='Neighborhood'

plt.figure(figsize=(25,10))

plt.subplot(211)

combined[col].value_counts().plot.bar()

plt.subplot(212)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)



ind = combined['BrkSide'].loc[combined['BrkSide']==1].index

combined.loc[ind,'OldTown'] = 1

ind = combined['SWISU'].loc[combined['SWISU']==1].index

combined.loc[ind,'Sawyer'] = 1

ind = combined['NPkVill'].loc[combined['NPkVill']==1].index

combined.loc[ind,'Blueste'] = 1

ind = combined['SawyerW'].loc[combined['SawyerW']==1].index

combined.loc[ind,'Crawfor'] = 1

ind1 = combined['Mitchel'].loc[combined['Mitchel']==1].index

ind2 = combined['Gilbert'].loc[combined['Gilbert']==1].index

combined.loc[ind1,'NWAmes']=1

combined.loc[ind2,'NWAmes']=1

ind3 = combined['NoRidge'].loc[combined['NoRidge']==1].index

ind4 = combined['NridgHt'].loc[combined['NridgHt']==1].index

combined.loc[ind3,'StoneBr']=1

combined.loc[ind4,'StoneBr']=1

ind5 = combined['IDOTRR'].loc[combined['IDOTRR']==1].index

ind6 = combined['MeadowV'].loc[combined['MeadowV']==1].index

combined.loc[ind5,'BrDale']=1

combined.loc[ind6,'BrDale']=1

print(combined['Timber'].loc[combined['Timber']==1].shape)

ind7 = combined['Somerst'].loc[combined['Somerst']==1].index

ind8 = combined['ClearCr'].loc[combined['ClearCr']==1].index

ind9 = combined['Veenker'].loc[combined['Veenker']==1].index

combined.loc[ind7,'Timber']=1

combined.loc[ind8,'Timber']=1

combined.loc[ind9,'Timber']=1

print(combined['Timber'].loc[combined['Timber']==1].shape)

combined = combined.drop(['Neighborhood','Sawyer','BrkSide','SWISU','Gilbert','Mitchel','Somerst','ClearCr','Veenker','NoRidge','NridgHt','IDOTRR','MeadowV','NPkVill'],axis=1)
col='Condition2'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
#ind = combined['Condition2'].loc[combined['Condition2']=='Norm'].index

#print(ind)

combined = combined.drop(['Condition2','Condition1'],axis=1)
col='BldgType'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
print(combined[col].value_counts())

df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

ind = combined['Duplex'].loc[combined['Duplex']==1].index

combined.loc[ind,'2fmCon'] = 1

combined = combined.drop(['Duplex','TwnhsE','Twnhs','BldgType'],axis=1)

col='HouseStyle'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['SLvl','SFoyer','2.5Unf','1.5Unf','2.5Unf','HouseStyle'],axis=1)
col='OverallQual'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

combined['oqlast3'] = combined[3]

combined['oqtop2'] = combined[9]

print(combined['oqlast3'].loc[combined['oqlast3']==1].shape)

ind1 = combined[1].loc[combined[1]==1].index

ind2 = combined[2].loc[combined[2]==1].index

combined.loc[ind1,'oqlast3'] = 1

combined.loc[ind2,'oqlast3'] = 1



print(combined['oqlast3'].loc[combined['oqlast3']==1].shape)

print(combined['oqtop2'].loc[combined['oqtop2']==1].shape)

ind1 = combined[10].loc[combined[10]==1].index

combined.loc[ind1,'oqtop2'] = 1

print(combined['oqtop2'].loc[combined['oqtop2']==1].shape)

print(ind1)

combined['oq4'] = combined[4]

combined['oq5'] = combined[5]

combined['oq6'] = combined[6]

combined['oq7'] = combined[7]

combined['oq8'] = combined[8]



combined = combined.drop(['OverallQual',1,2,3,4,5,6,7,8,9,10],axis=1)
col='OverallCond'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

combined['oc5'] = combined[5]

combined['oc6'] = combined[6]

print(combined['oc6'].loc[combined['oc6']==1].shape)

ind = combined[7].loc[combined[7]==1].index

combined.loc[ind,'oc6'] = 1

print(combined['oc6'].loc[combined['oc6']==1].shape)

ind = combined[8].loc[combined[8]==1].index

combined.loc[ind,'oc6'] = 1

print(combined['oc6'].loc[combined['oc6']==1].shape)

combined = combined.drop([1,2,3,4,9,'OverallCond',7,8,5,6],axis=1)
col='YearBuilt'

plt.figure(figsize=(25,10))

plt.subplot(211)

combined[col].value_counts().plot.bar()

plt.subplot(212)

sns.violinplot(train_X[col],train_y)

plt.show()
labels=['very_old','old','avg','new']

cut_pts = [1950,1980,1999]

combined['Year_bins'] = binning(combined['YearBuilt'],cut_pts,labels)

train_X['Year_bins'] = binning(train_X['YearBuilt'],cut_pts,labels)
col='Year_bins'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined['Year_bins'])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['YearBuilt','Year_bins'],axis=1)
col='YearRemodAdd'

plt.figure(figsize=(25,10))

plt.subplot(211)

combined[col].value_counts().plot.bar()

plt.subplot(212)

sns.violinplot(train_X[col],train_y)

plt.show()
labels=['very_old_radd','old_radd','avg_radd','new_radd']

cut_pts = [1970,1980,2002]

combined['modyear_bins'] = binning(combined['YearRemodAdd'],cut_pts,labels)

train_X['modyear_bins'] = binning(train_X['YearRemodAdd'],cut_pts,labels)
col='modyear_bins'

plt.figure(figsize=(25,10))

plt.subplot(211)

combined[col].value_counts().plot.bar()

plt.subplot(212)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined['modyear_bins'])

combined = pd.concat([combined,df],axis=1)

print(combined['old_radd'].loc[combined['old_radd']==1].shape)

ind = combined['very_old_radd'].loc[combined['very_old_radd']==1].index

combined.loc[ind,'old_radd'] = 1

print(combined['old_radd'].loc[combined['old_radd']==1].shape)

print(combined['new_radd'].loc[combined['new_radd']==1].shape)

ind = combined['avg_radd'].loc[combined['avg_radd']==1].index

combined.loc[ind,'new_radd'] = 1

print(combined['new_radd'].loc[combined['new_radd']==1].shape)



combined = combined.drop(['YearBuilt','very_old_radd','avg_radd','modyear_bins','YearRemodAdd'],axis=1)
col='RoofStyle'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined[col])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['Gambrel','Mansard','Flat','RoofStyle','Shed'],axis=1)

col = ['RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtExposure']

count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(3,3,count)

    combined[i].value_counts().plot.bar()

plt.show()

    

    
count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(3,3,count)

    sns.violinplot(train_X[i],train_y)

plt.show()

    
combined = combined.drop(['RoofMatl'],axis=1)

df = pd.get_dummies(combined['Exterior1st'])

combined = pd.concat([combined,df],axis=1)



combined = combined.drop(['CemntBd','BrkFace','WdShing','AsbShng','Stucco','BrkComm','CBlock','Stone','AsphShn','none','ImStucc','Exterior1st'],axis=1)



col = ['MasVnrType','ExterQual','Foundation']

for i in col :

    df = pd.get_dummies(combined[i])

    combined = pd.concat([combined,df],axis=1)

ind = combined['BrkCmn'].loc[combined['BrkCmn']==1].index

combined.loc[ind,'None'] = 1

combined = combined.drop(['BrkCmn','MasVnrType','ExterCond'],axis=1)

combined = combined.drop(['Stab','Stone','Wood','Foundation','ExterQual','BsmtQual','BsmtExposure'],axis=1)



col = ['BsmtFinType1','BsmtFinType2','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold','YrSold']

count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(3,3,count)

    sns.violinplot(train_X[i],train_y)

plt.show()
col = ['BsmtFinType1','BsmtFinType2','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold','YrSold']

count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(3,3,count)

    combined[i].value_counts().plot.bar()

plt.show()
col = ['BsmtFinType1','PavedDrive','PoolQC','Fence','MoSold','YrSold']

for i in col :

    df = pd.get_dummies(combined[i])

    combined = pd.concat([combined,df],axis=1)

    

ind = combined['BLQ'].loc[combined['BLQ']==1].index

combined.loc[ind,'Rec'] = 1

ind = combined['LwQ'].loc[combined['LwQ']==1].index

combined.loc[ind,'Rec'] = 1



combined['ispeakmonth'] = 0

ind = combined[8].loc[combined[8]==1].index

combined.loc[ind,'ispeakmonth'] = 1

ind = combined[9].loc[combined[9]==1].index

combined.loc[ind,'ispeakmonth'] = 1

ind = combined[11].loc[combined[11]==1].index

combined.loc[ind,'ispeakmonth'] = 1

ind = combined[12].loc[combined[12]==1].index

combined.loc[ind,'ispeakmonth'] = 1



combined = combined.drop(['BsmtFinType2','GarageCond','P','MiscFeature',1,2,3,4,5,6,7,8,9,10,11,12],axis=1)

combined = combined.drop(['BsmtFinType1','PavedDrive','PoolQC','Fence','MoSold','YrSold'],axis=1)
col = ['SaleType','SaleCondition','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu']

count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(3,3,count)

    combined[i].value_counts().plot.bar()

plt.show()

count=0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(3,3,count)

    sns.violinplot(train_X[i],train_y)

plt.show()
col = ['SaleType','SaleCondition','HeatingQC','CentralAir','Electrical','KitchenQual','Functional']

for i in col :

    df = pd.get_dummies(combined[i])

    combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth','none'],axis=1)

combined = combined.drop(['AdjLand', 'Alloca', 'Family','Heating'],axis=1)

combined = combined.drop(['Po'],axis=1)

combined['isSBrkr'] = 0

ind = combined['SBrkr'].loc[combined['SBrkr']==1].index

combined.loc[ind,'isSBrkr'] = 1

combined = combined.drop(['FuseF', 'FuseA', 'FuseP', 'Mix', 'none'],axis=1)



combined['isTyp'] = 0

ind = combined['Typ'].loc[combined['Typ']==1].index

combined.loc[ind,'isTyp'] = 1

combined = combined.drop(['Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev','FireplaceQu'],axis=1)

combined = combined.drop(['SaleType','SaleCondition','HeatingQC','CentralAir','Electrical','KitchenQual','Functional'],axis=1)

print(combined['isTyp'].value_counts())

print(combined['isSBrkr'].unique())
col = ['GarageType','GarageCars','GarageFinish','GarageQual']

count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(2,2,count)

    combined[i].value_counts().plot.bar()

    plt.xlabel(i)

plt.show()
col = ['GarageType','GarageCars','GarageFinish','GarageQual']

count = 0

plt.figure(figsize=(25,20))

for i in col :

    count = count+1

    plt.subplot(2,2,count)

    sns.violinplot(train_X[i],train_y)

plt.show()
col = ['GarageType','GarageCars','GarageFinish']

for i in col :

    df = pd.get_dummies(combined[i])

    combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['CarPort', 'none', 'Basment','2Types','GarageQual'],axis=1)

combined['Cars_0'] = combined[0]

combined['Cars_1'] = combined[1]

combined['Cars_2'] = combined[2]

combined['Cars_3'] = combined[3]

combined['Cars_4'] = combined[4]



combined = combined.drop(['GarageType','GarageCars','GarageFinish'],axis=1)

print(combined['Cars_4'].value_counts())
col='GarageYrBlt'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
ind = combined['GarageYrBlt'].loc[combined['GarageYrBlt']=='none'].index

print(ind)

combined.loc[ind,'GarageYrBlt'] = 1925

combined['GarageYrBlt'].unique()
labels = ['oldgar','avggar','newgar']

cut_pts = [1950,1990]

combined['Garage_blt_bins'] = binning(combined['GarageYrBlt'],cut_pts,labels)

train_X['Garage_blt_bins'] = binning(train_X['GarageYrBlt'],cut_pts,labels)
combined = combined.drop(['GarageYrBlt'],axis=1)
col='Garage_blt_bins'

plt.figure(figsize=(25,10))

plt.subplot(121)

combined[col].value_counts().plot.bar()

plt.subplot(122)

sns.violinplot(train_X[col],train_y)

plt.show()
df = pd.get_dummies(combined['Garage_blt_bins'])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['Garage_blt_bins'],axis=1)
col='Exterior2nd'

plt.figure(figsize=(25,10))

plt.subplot(211)

combined[col].value_counts().plot.bar()

plt.subplot(212)

sns.violinplot(train_X[col],train_y)

plt.show()
combined = combined.drop(['Exterior2nd'],axis=1)
col='BsmtCond'

plt.figure(figsize=(25,10))

plt.subplot(211)

combined[col].value_counts().plot.bar()

plt.subplot(212)

sns.violinplot(train_X[col],train_y)

plt.show()
combined = combined.drop(['BsmtCond','None'],axis=1)
print(combined.shape)

print(train_X.shape)

print(test.shape)

combined = combined.drop(['none'],axis=1)
col = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageArea']

for i in col :

    combined[i] = combined[i].replace(['none'],[0])



    

combined.columns


combined.to_csv('check',index=False)

combined['LotFrontage']
print(combined.shape)

train_data_X = combined.iloc[:1460,:]

test_data = (combined.iloc[1460:,:]).reset_index()

print(train_data_X.index)

print(test_data.index)

print(train_data_X.shape)

print(test_data.shape)

test_data = test_data.drop(['index'],axis=1)
print(train_data_X.head(5))

print(test_data.head(5))

test_data = test_data.drop(['index'],axis=1)




from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

train_data_X = scale(train_data_X)

X_train,X_test,y_train,y_test=train_test_split(train_data_X,train_y,test_size=0.5,random_state=42)

reg = linear_model.Lasso(max_iter=100)

reg.fit(X_train,y_train)

test_data = scale(test_data)

y_pred = reg.predict(test_data)

sub = pd.DataFrame({'Id':Hid,'SalePrice':y_pred})

sub.to_csv('sub1',index=False)

scores = cross_val_score(reg,X_test,y_test,cv=10)

print(reg.score(X_test,y_test))

print(scores)
sub['SalePrice'].loc[sub['Id']==1975]
sub.head()