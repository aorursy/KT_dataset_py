import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

# Load the train data in a dataframe

train = pd.read_csv("../input/train.csv")



# Load the test data in a dataframe

test = pd.read_csv("../input/test.csv")
# Look at the head of the train dataframe

train.head()
train.info()
# Look at the SalePrice Variable

sns.distplot(train.SalePrice)
train.SalePrice.describe()
nulls = train.isnull().sum().sort_values(ascending=False)

nulls.head(20)
train = train.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)
train[['Fireplaces','FireplaceQu']].head(10)
train['FireplaceQu'].isnull().sum()
train['Fireplaces'].value_counts()
# I used fillna() to replace the nulls with NF

train['FireplaceQu']=train['FireplaceQu'].fillna('NF')
#LotFrontage

train['LotFrontage'] =train['LotFrontage'].fillna(value=train['LotFrontage'].mean())
train['GarageType'].isnull().sum()
train['GarageCond'].isnull().sum()
train['GarageFinish'].isnull().sum()
train['GarageYrBlt'].isnull().sum()
train['GarageQual'].isnull().sum()
train['GarageArea'].value_counts().head()
train['GarageType']=train['GarageType'].fillna('NG')

train['GarageCond']=train['GarageCond'].fillna('NG')

train['GarageFinish']=train['GarageFinish'].fillna('NG')

train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')

train['GarageQual']=train['GarageQual'].fillna('NG')
#Similarly for the attributes of Bsmt I'll replace with NB

train['BsmtExposure']=train['BsmtExposure'].fillna('NB')

train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')

train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')

train['BsmtCond']=train['BsmtCond'].fillna('NB')

train['BsmtQual']=train['BsmtQual'].fillna('NB')
# MasVnr

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

train['MasVnrType'] = train['MasVnrType'].fillna('none')
train.Electrical = train.Electrical.fillna('SBrkr')
#confirm that the train doesn't have any null values

train.isnull().sum().sum()
num_train = train._get_numeric_data()
# I'll write a pre defined function to look into the outliars with the help of percentiles.

def var_summary(x):

    return pd.Series([x.mean(), x.median(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index=['MEAN','MEDIAN', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_train.apply(lambda x: var_summary(x)).T

# Boxplot

sns.boxplot(train.LotArea)
# Clipping the values which are above 95-pecentile

train['LotArea']= train['LotArea'].clip_upper(train['LotArea'].quantile(0.95)) 
sns.boxplot(train.MasVnrArea)
train['MasVnrArea']= train['MasVnrArea'].clip_upper(train['MasVnrArea'].quantile(0.95)) 
sns.boxplot(train.BsmtFinSF1)
train['BsmtFinSF1']= train['BsmtFinSF1'].clip_upper(train['BsmtFinSF1'].quantile(0.95)) 
sns.boxplot(train.BsmtFinSF2)
train['BsmtFinSF2']= train['BsmtFinSF2'].clip_upper(train['BsmtFinSF2'].quantile(0.99)) 
train['BsmtUnfSF']= train['BsmtUnfSF'].clip_upper(train['BsmtUnfSF'].quantile(0.99)) 

train['TotalBsmtSF']= train['TotalBsmtSF'].clip_upper(train['TotalBsmtSF'].quantile(0.99)) 

train['1stFlrSF']= train['1stFlrSF'].clip_upper(train['1stFlrSF'].quantile(0.99)) 

train['2ndFlrSF']= train['2ndFlrSF'].clip_upper(train['2ndFlrSF'].quantile(0.99)) 

train['LowQualFinSF']= train['LowQualFinSF'].clip_upper(train['LowQualFinSF'].quantile(0.99)) 

train['GrLivArea']= train['GrLivArea'].clip_upper(train['GrLivArea'].quantile(0.99)) 

train['PoolArea']= train['PoolArea'].clip_upper(train['PoolArea'].quantile(0.99)) 

train['MiscVal']= train['MiscVal'].clip_upper(train['MiscVal'].quantile(0.99)) 



sns.boxplot(train.SalePrice)
train['SalePrice']= train['SalePrice'].clip_upper(train['SalePrice'].quantile(0.99)) 
# CORREALATION

correlation = num_train .corr()

plt.subplots(figsize=(10,10))

sns.heatmap(correlation,vmax =.8 ,square = True)
# Look for highly correlated variables

k = 12

cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(num_train[cols].values.T)

sns.set(font_scale=1.35)

f, ax = plt.subplots(figsize=(10,8))

hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values)
# Dummifyiny the categorical data

dum_train = pd.get_dummies(train)
from sklearn.ensemble import RandomForestRegressor
dum_train_x = dum_train.drop(["SalePrice"],axis = 1)

dum_train_y = dum_train.SalePrice



X_train = dum_train_x

Y_train = dum_train_y
radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100 )

radm_clf.fit( X_train, Y_train )
indices = np.argsort(radm_clf.feature_importances_)[::-1]
indices = np.argsort(radm_clf.feature_importances_)[::-1]

feature_rank = pd.DataFrame( columns = ['rank', 'feature', 'importance'] )

for f in range(X_train.shape[1]):

    feature_rank.loc[f] = [f+1,

                         X_train.columns[indices[f]],

                         radm_clf.feature_importances_[indices[f]]]

f, ax = plt.subplots(figsize=(10,100))

sns.barplot( y = 'feature', x = 'importance', data = feature_rank, color = 'Yellow')

plt.show()
best_train = feature_rank.head(50)

best_train