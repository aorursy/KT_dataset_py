import numpy as np 

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.model_selection import cross_val_score

from scipy.stats import norm

from scipy import stats 

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train= pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')



print("shape of train dataset:", train.shape)

print("shape of test dataset:", test.shape)
train.head()
plt.figure(figsize=[10,6])

sns.distplot(train.SalePrice, fit=norm)
plt.figure(figsize=[10,6])

sns.distplot(np.log(train.SalePrice), fit=norm)
correlation= train.corr()

plt.figure(figsize=[12,8])

plt.title('Correlation of Numeric Features with Sale Price')

sns.heatmap(correlation,cmap="YlGnBu")
correlation= train.corr()

correlation=correlation['SalePrice'].sort_values(ascending=False)

pos_correlation=correlation.head(25)

pos_correlation
plt.figure(figsize=[8,5])

sns.regplot(train['OverallQual'], train['SalePrice'])
plt.figure(figsize=[8,5])

sns.regplot(train['GrLivArea'], train['SalePrice'])
plt.figure(figsize=[8,5])

sns.regplot(train['GarageArea'], train['SalePrice'])
plt.figure(figsize=[8,5])

sns.regplot(train['TotalBsmtSF'], train['SalePrice'])
print("Before:",train.shape)



train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)

train.drop(train[(train['GarageArea']>1200) & (train['SalePrice']<100000)].index, inplace=True)

train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index, inplace=True)

train.reset_index(drop=True, inplace=True)



print("After:",train.shape)
data= pd.concat([train.drop(['SalePrice'], axis=1), test])
xx= (data.isnull().sum())/len(data)*100

xx=xx.sort_values(ascending=False).head(30)



plt.figure(figsize=(15, 6))

plt.xticks(rotation="90")

sns.barplot(xx.keys(), xx)
total=data.isnull().sum().sort_values(ascending=False)

percent=((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)

missing= pd.concat([total,percent], axis=1, join='outer', keys=['Total missing count', 'Percentage '])

missing.head(30)
train['GarageQual'].fillna('None', inplace=True)

test['GarageQual'].fillna('None', inplace=True)

train['GarageFinish'].fillna('None', inplace=True)

test['GarageFinish'].fillna('None', inplace=True)

train['GarageYrBlt'].fillna('None', inplace=True)

test['GarageYrBlt'].fillna('None', inplace=True)

train['GarageType'].fillna('None', inplace=True)

test['GarageType'].fillna('None', inplace=True)

train['GarageCond'].fillna('None', inplace=True)

test['GarageCond'].fillna('None', inplace=True)

test.loc[test['Id']==2577, 'GarageType']='None'



test['GarageCars'].fillna(0, inplace=True)

test['GarageArea'].fillna(0, inplace=True)
data[(data['TotalBsmtSF']!=0) & (data['BsmtCond'].isnull()==True)][['Id','TotalBsmtSF','BsmtCond','BsmtQual','BsmtExposure',

                                                                    'BsmtFinType1','BsmtFinSF1','BsmtUnfSF']]
test.loc[test['Id']==2041, 'BsmtCond']='Gd'

test.loc[test['Id']==2186, 'BsmtCond']='TA'

test.loc[test['Id']==2525, 'BsmtCond']='TA'

train['BsmtCond'].fillna('None', inplace=True)

test['BsmtCond'].fillna('None', inplace=True)
data[(data['TotalBsmtSF']!=0) & (data['BsmtExposure'].isnull()==True)][['Id','TotalBsmtSF','BsmtCond','BsmtQual','BsmtExposure',

                                                                    'BsmtFinType1','BsmtFinSF1','BsmtUnfSF']]
train.loc[train['Id']==949, 'BsmtExposure']='Gd'

test.loc[test['Id']==1488, 'BsmtExposure']='Gd'

test.loc[test['Id']==2349, 'BsmtExposure']='Gd'

train['BsmtExposure'].fillna('None', inplace=True)

test['BsmtExposure'].fillna('None', inplace=True)
data[(data['TotalBsmtSF']!=0) & (data['BsmtQual'].isnull()==True)][['Id','TotalBsmtSF','BsmtQual','BsmtCond','BsmtExposure',

                                                                    'BsmtFinType1','BsmtFinSF1','BsmtUnfSF']]
test.loc[test['Id']==2218, 'BsmtQual']='Fa'

test.loc[test['Id']==2219, 'BsmtQual']='TA'

train['BsmtQual'].fillna('None', inplace=True)

test['BsmtQual'].fillna('None', inplace=True)
data[(data['TotalBsmtSF']!=0) & (data['BsmtFinType1'].isnull()==True)][['Id','TotalBsmtSF','BsmtQual','BsmtCond','BsmtExposure',

                                                                        'BsmtFinType1','BsmtFinSF1','BsmtUnfSF']]
data[(data['TotalBsmtSF']!=0) & (data['BsmtFullBath'].isnull()==True)][['Id','TotalBsmtSF','BsmtQual','BsmtCond',

                                                                        'BsmtExposure','BsmtFullBath']]
train['BsmtFinType1'].fillna('None', inplace=True)

test['BsmtFinType1'].fillna('None', inplace=True)

test['BsmtFinSF1'].fillna(0, inplace=True)

train['BsmtFinType2'].fillna('None', inplace=True)

test['BsmtFinType2'].fillna('None', inplace=True)

test['BsmtUnfSF'].fillna(0, inplace=True)

test['TotalBsmtSF'].fillna(0, inplace=True)



test['BsmtFullBath'].fillna(0, inplace=True)

test['BsmtHalfBath'].fillna(0, inplace=True)



test['BsmtFinSF1'].fillna(0, inplace=True)

test['BsmtFinSF2'].fillna(0, inplace=True)
train['FireplaceQu'].fillna('None', inplace=True)

test['FireplaceQu'].fillna('None', inplace=True)



train.loc[train['LotFrontage'].isnull()==True, 'LotFrontage']= train['LotFrontage'].mean()

test.loc[test['LotFrontage'].isnull()==True, 'LotFrontage']= test['LotFrontage'].mean()



train['MasVnrType'].fillna('None', inplace=True)

test['MasVnrType'].fillna('None', inplace=True)



train['MasVnrArea'].fillna(0, inplace=True)

test['MasVnrArea'].fillna(0, inplace=True)



train['PoolQC'].fillna('None', inplace=True)

test['PoolQC'].fillna('None', inplace=True)



train['MiscFeature'].fillna('None', inplace=True)

test['MiscFeature'].fillna('None', inplace=True)



train['Alley'].fillna('None', inplace=True)

test['Alley'].fillna('None', inplace=True)



train['Fence'].fillna('None', inplace=True)

test['Fence'].fillna('None', inplace=True)



train['MSZoning'].fillna('RL', inplace=True)

test['MSZoning'].fillna('RL', inplace=True)



train['Electrical'].fillna('SBrkr', inplace=True)

test['Functional'].fillna('Typ', inplace=True)

test['SaleType'].fillna('WD', inplace=True)

test['Exterior1st'].fillna('VinylSd', inplace=True)

test['KitchenQual'].fillna('TA', inplace=True)

test['Exterior2nd'].fillna('VinylSd', inplace=True)
train.drop('Utilities', axis=1, inplace=True)

test.drop('Utilities', axis=1, inplace=True)
print("Train dataset:\n",train.isnull().sum().sort_values(ascending=False).head(5))

print("\n\nTest dataset:\n",test.isnull().sum().sort_values(ascending=False).head(5))
train['renovated']= 1

train.loc[(train['YearBuilt']!=train['YearRemodAdd']),'renovated' ]=0

test['renovated']= 1

test.loc[(test['YearBuilt']!=test['YearRemodAdd']),'renovated' ]=0



train['Total_porch']= (train['OpenPorchSF'])+train['ScreenPorch']+train['3SsnPorch']

train['BsmtFinSF_total']= (train['BsmtFinSF1']**2)+train['BsmtFinSF2']

train['Bath']= (train['FullBath']**2)+ train['HalfBath']

train['BsmtBath']= (train['BsmtFullBath'])+train['BsmtHalfBath']

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

train['Total_garage']= (train['GarageCars']**5)+ train['GarageArea']





test['Total_porch']= (test['OpenPorchSF'])+test['ScreenPorch']+test['3SsnPorch']

test['BsmtFinSF_total']= (test['BsmtFinSF1']**2)+test['BsmtFinSF2']

test['Bath']= (test['FullBath']**2)+ test['HalfBath']

test['BsmtBath']= (test['BsmtFullBath'])+test['BsmtHalfBath']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

test['Total_garage']= (test['GarageCars']**5)+ test['GarageArea']
y=np.asarray(train['SalePrice'])

y=np.log(y+1)



X_train = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1),

                                          test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

X_test = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1),

                                         test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]



X_train.shape, X_test.shape
def return_rmse(model):

    return np.sqrt(-cross_val_score(model, X_train, y, cv=5, scoring="neg_mean_squared_error")).mean()
RR= Ridge(alpha=15)

return_rmse(RR)
LSR = Lasso(alpha=0.0005)

return_rmse(LSR)
EN = ElasticNet(alpha=0.01,l1_ratio=0.1)

return_rmse(EN)
RR.fit(X_train, y)

LSR.fit(X_train, y)

EN.fit(X_train, y)



y_pred = RR.predict(X_train)

residual = y - y_pred

z = np.abs(stats.zscore(residual))

outliers1=np.where(abs(z) > abs(z).std() * 3)[0]

outliers1
y_pred = LSR.predict(X_train)

residual = y - y_pred

z = np.abs(stats.zscore(residual))

outliers2=np.where(abs(z) > abs(z).std() * 3)[0]

outliers2
y_pred = EN.predict(X_train)

residual = y - y_pred

z = np.abs(stats.zscore(residual))

outliers3=list(np.where(abs(z) > abs(z).std() * 3))[0]

outliers3
outliers = []

for i in outliers1:

    if (i in outliers2) & (i in outliers3):

        outliers.append(i)     



train = train.drop(outliers)        
y = train["SalePrice"]

y = np.log(y+1)



X_train = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1),

                                    test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

X_test = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1),

                                   test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]
RR= Ridge(alpha=15)

return_rmse(RR)
LSR = Lasso(alpha=0.0004)

return_rmse(LSR)
EN = ElasticNet(alpha=0.001,l1_ratio=0.5)

return_rmse(EN)
RR.fit(X_train, y)

LSR.fit(X_train, y)

EN.fit(X_train, y)



predict = 0.4 * RR.predict(X_test) + 0.3 * EN.predict(X_test) + 0.3 * LSR.predict(X_test)



predict= np.exp(predict)-1

sample_submission= pd.DataFrame({'Id':np.asarray(test.Id), 'SalePrice':predict})

sample_submission.to_csv("submit.csv", index=False)