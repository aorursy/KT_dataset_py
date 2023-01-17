import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

Hid = test['Id']
train_y = train['SalePrice']

train_X = train.drop(['SalePrice'],axis=1)

frames = [train_X,test]

combined = pd.concat(frames,ignore_index=True)

col = combined.columns
print(combined.shape)

combined.head()
combined.isnull().sum()
combined = combined.drop(['PoolQC','PoolArea'],axis=1)
combined['no_alley']=0

ind = combined.loc[combined['Alley'].isnull()].index

combined.loc[ind,'no_alley'] = 1
combined['no_alley'].value_counts()
combined['Fence'].value_counts()
plt.figure(figsize=(20,10))

t1 = list(train[train['Fence'].notnull()==True].index)

t2 = list(train[train['Fence'].isnull()==True].index)

t3 = list(train[train['Fence'] == 'MnPrv'].index)

t4 = list(train[train['Fence'] == 'GdPrv'].index)

t5 = list(train[train['Fence'] == 'GdWo'].index)

t6 = list(train[train['Fence'] == 'MnWw'].index)

plt.scatter(t2,train_y.loc[t2],marker='x',c='blue',alpha=0.3)

plt.scatter(t3,train_y.loc[t3],marker='x')

plt.scatter(t4,train_y.loc[t4],marker='x')

plt.scatter(t5,train_y.loc[t5],marker='x')

plt.scatter(t6,train_y.loc[t6],marker='x')



plt.show()
t1 = list(combined[combined['Fence'].notnull()==True].index)

combined['Has_Fence'] = 0

combined['Has_Fence'].loc[t1] = 1

combined['Has_Fence'].value_counts()
plt.figure(figsize=(20,10))

combined['MiscFeature'].value_counts()

t1 = list(train[train['MiscFeature'].notnull()==True].index)

t2 = list(train[train['MiscFeature'].isnull()==True].index)

plt.scatter(t2,train_y.loc[t2],marker='x',c='blue',alpha=0.3)

plt.scatter(t1,train_y.loc[t1],marker='X',alpha=1,c='red')

plt.show()
t1 = list(combined[combined['MiscFeature'].notnull()==True].index)

combined['Has_MiscFea'] = 0

combined['Has_MiscFea'].loc[t1] = 1

combined['Has_MiscFea'].value_counts()
combined = combined.drop(['Fence','MiscFeature'],axis=1)
print(combined['GarageType'].value_counts())

t1 = list(combined[combined['GarageType'].notnull()==True].index)
df = pd.get_dummies(combined['GarageType'])



combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['GarageType'],axis=1)

df.head()
df = pd.get_dummies(combined['GarageFinish'])



combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['GarageFinish'],axis=1)

df.head()
combined['GarageCars'].value_counts()
combined['GarageCars'] = combined['GarageCars'].replace([1,2,3,4,5,0],['cars_1','cars_2','cars_3','cars_4','cars_5','no_cars'])

df = pd.get_dummies(combined['GarageCars'])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['GarageCars'],axis=1)

df.head()
plt.scatter(train_X['GarageArea'],train_y,s=9)

plt.show()
combined.loc[combined['GarageArea']==0,'GarageArea'].shape

combined['GarageArea'] = combined['GarageArea'].fillna(combined['GarageArea'].mean())

plt.hist(combined['GarageArea'],bins=50)

plt.show()

combined['GarageQual'].value_counts()
combined['GarageCond'].value_counts()
combined['Fireplaces'].value_counts()

combined['Fireplaces'] = combined['Fireplaces'].replace([1,2,3,4,0],['1_fp','2_fp','3_fp','4_fp','no_fp'])
combined['FireplaceQu'].value_counts()
combined['KitchenQual'].value_counts()
print(combined['MasVnrArea'].value_counts())

print(combined['MasVnrType'].value_counts())
combined.loc[combined['MasVnrType'].isnull()].loc[combined['MasVnrArea'].isnull()]
combined.loc[combined['MasVnrType']=='None','MasVnrArea'] = 0

print(combined.loc[combined['MasVnrArea'].isnull()==True].shape)

combined['MasVnrArea'] = combined['MasVnrArea'].fillna(combined['MasVnrArea'].mean())

combined['MasVnrArea'].value_counts()
combined['LotFrontage'].value_counts()

plt.scatter(combined['LotArea'],combined['LotFrontage'],s=9)

plt.show()
ulimit = np.percentile(combined['LotArea'].values,99)

llimit = np.percentile(combined['LotArea'].values,2)

combined.loc[combined['LotArea']>ulimit,'LotArea']=ulimit

combined.loc[combined['LotArea']<llimit,'LotArea']=llimit
indx = combined['LotFrontage'].loc[combined['LotFrontage'].isnull()].index

combined.loc[indx,'LotFrontage'] = combined['LotArea'].loc[indx]/100
combined['LotFrontage'].loc[combined['LotFrontage'].isnull()]
combined = pd.get_dummies(combined)
t = combined.isnull().sum() 

t[t!=0]
combined['BsmtFinSF2'].value_counts()
plt.scatter(combined['BsmtFinSF1'],combined['BsmtFinSF2'],s=9)

plt.xlabel('BsmtFinSF1')

plt.ylabel('BsmtFinSF2')

plt.show()
combined['TotalBsmt_Fin_SF'] = combined['BsmtFinSF1'] + combined['BsmtFinSF2']

combined['TotalBsmt_Fin_SF'].head(5)
combined['TotalBsmtSF'].head(5)
combined = combined.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis=1)
combined['BsmtFullBath'] = combined['BsmtFullBath'].fillna(0)

combined['BsmtHalfBath'] = combined['BsmtHalfBath'].fillna(0)

combined['TotalBsmtSF'] = combined['TotalBsmtSF'].fillna(0)

combined['TotalBsmt_Fin_SF'] = combined['TotalBsmt_Fin_SF'].fillna(0)



print(combined['BsmtFullBath'].value_counts())

print(combined['BsmtHalfBath'].value_counts())

combined['BsmtHalfBath'].unique()
combined['Bsmt_Bath'] = 0

names = ['BsmtBath_cat1','BsmtBath_cat2','BsmtBath_cat3','BsmtBath_cat4','BsmtBath_cat5','BsmtBath_cat6','BsmtBath_cat7','BsmtBath_cat8','BsmtBath_cat9','BsmtBath_cat10','BsmtBath_cat11','BsmtBath_cat12']

count=0

for i in combined['BsmtFullBath'].unique():

    for j in combined['BsmtHalfBath'].unique():

        indx = (combined['Bsmt_Bath'].loc[combined['BsmtFullBath']==i].loc[combined['BsmtHalfBath']==j]).index

        print(indx)

        combined.loc[indx,'Bsmt_Bath'] = names[count]

        count = count+1

        
combined['Bsmt_Bath'].value_counts()
df = pd.get_dummies(combined['Bsmt_Bath'])

combined = pd.concat([combined,df],axis=1)

df.head()

combined = combined.drop(['BsmtBath_cat3','BsmtBath_cat10','BsmtBath_cat6','BsmtFullBath','BsmtHalfBath','Bsmt_Bath'],axis=1)
def binning(col,cut_pts,labels=None

            ):

    minval=col.min()

    maxval=col.max()

    break_pts = [minval]+cut_pts+[maxval]

    if not labels:

        labels = range(len(cut_pts)+1)

    col_bins=pd.cut(col,bins=break_pts,labels=labels,include_lowest=True)

    return col_bins
combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(0.1)

labels = ['No_Garage','oldgar','avggar','newgar']

cut_pts = [1,1950,1990]

combined['Garage_blt_bins'] = binning(combined['GarageYrBlt'],cut_pts,labels)
df = pd.get_dummies(combined['GarageYrBlt'])

combined = pd.concat([combined,df],axis=1)

df.head()

combined = combined.drop(['GarageYrBlt','Garage_blt_bins'],axis=1)
t = combined.isnull().sum() 

t[t!=0]
labels=['very_old','old','avg','new']

cut_pts = [1950,1980,1999]

combined['Year_bins'] = binning(combined['YearBuilt'],cut_pts,labels)
df = pd.get_dummies(combined['Year_bins'])

combined = pd.concat([combined,df],axis=1)

combined = combined.drop(['YearBuilt','Year_bins'],axis=1)
labels=['very_old_radd','old_radd','avg_radd','new_radd']

cut_pts = [1970,1980,2002]

combined['modyear_bins'] = binning(combined['YearRemodAdd'],cut_pts,labels)
df = pd.get_dummies(combined['modyear_bins'])

combined = pd.concat([combined,df],axis=1)



ind = combined['very_old_radd'].loc[combined['very_old_radd']==1].index

combined.loc[ind,'old_radd'] = 1



ind = combined['avg_radd'].loc[combined['avg_radd']==1].index

combined.loc[ind,'new_radd'] = 1



combined = combined.drop(['very_old_radd','avg_radd','modyear_bins','YearRemodAdd'],axis=1)
combined['YrSold'] = combined['YrSold'].replace([2006,2007,2008,2009,2010],['SaleYr_06','SaleYr_07','SaleYr_08','SaleYr_09','SaleYr_10'])

df = pd.get_dummies(combined['YrSold'])

combined = pd.concat([combined,df],axis=1)
combined = combined.drop(['YrSold'],axis=1)
combined['MoSold'].value_counts()
combined['MoSold'] = combined['MoSold'].replace([1,2,3,4,5,6,7,8,9,10,11,12],['jan','feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec'])

df = pd.get_dummies(combined['MoSold'])

combined = pd.concat([combined,df],axis=1)
combined = combined.drop(['MoSold'],axis=1)
train_data_X = combined.iloc[:1460,:]

test_data = (combined.iloc[1460:,:]).reset_index()

print(train_data_X.index)

print(test_data.index)

print(train_data_X.shape)

print(test_data.shape)

test_data = test_data.drop(['index'],axis=1)

print(train_data_X.head())

print(test_data.head())
from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

train_data_X = scale(train_data_X)

X_train,X_test,y_train,y_test=train_test_split(train_data_X,train_y,test_size=0.5,random_state=42)

reg = linear_model.Lasso(alpha = 1000)

reg.fit(X_train,y_train)

test_data = scale(test_data)

y_pred = reg.predict(test_data)

sub = pd.DataFrame({'Id':Hid,'SalePrice':y_pred})

sub.to_csv('submt2',index=False)

scores = cross_val_score(reg,X_test,y_test,cv=10)

print(reg.score(X_test,y_test))

print(reg.score(X_train,y_train))

print(scores)

print(np.sum(reg.coef_!=0))
from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 1000)

ridge.fit(X_train,y_train)

print(ridge.score(X_test,y_test))

print(ridge.score(X_train,y_train))

scores = cross_val_score(ridge,X_test,y_test,cv=10)

print(scores)

#y_pred = ridge.predict(test_data)

#sub = pd.DataFrame({'Id':Hid,'SalePrice':y_pred})

#sub.to_csv('submt3',index=False)
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0,learning_rate=.2,max_depth=2)

gbrt.fit(X_train,y_train)

print(gbrt.score(X_test,y_test))

print(gbrt.score(X_train,y_train))

#y_pred = gbrt.predict(test_data)

#sub = pd.DataFrame({'Id':Hid,'SalePrice':y_pred})

#sub.to_csv('submt4',index=False)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=45,random_state=42)

rf.fit(X_train,y_train)

print(rf.score(X_test,y_test))

print(rf.score(X_train,y_train))

#y_pred = rf.predict(test_data)

#sub = pd.DataFrame({'Id':Hid,'SalePrice':y_pred})

#sub.to_csv('submt5',index=False)
reg.coef_
fi = pd.DataFrame({'feature_name':combined.columns,'imp_value':reg.coef_})

imp = fi.loc[fi['imp_value']>0]

plt.figure(figsize=(40,20))

plt.plot(imp.imp_value.values,'o',markersize=15)

plt.xticks(range(len(imp)),imp.feature_name.values,rotation=90,fontsize=25)

plt.show()