# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy import stats

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")#忽略掉普通的warning

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv",index_col=0)

test=pd.read_csv(("../input/test.csv"),index_col=0)

test['SalePrice']=-99
sns.distplot(np.log1p(train['SalePrice']),color="r")
na_des=train.isna().sum()

na_des[na_des>0].sort_values(ascending=False)
new_data=pd.concat([train,test],axis=0,sort=False)

new_data.head()
cols1 = ["PoolQC","MiscFeature",'SaleType', "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]

for col in cols1:

    new_data[col].fillna("None", inplace=True)

cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]

for col in cols:

    new_data[col].fillna(0, inplace=True)

new_data["LotFrontage"] = new_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

a=new_data.isna().sum()

cols=a[(a>0) & (a<100)].index

for col in cols:

    new_data[col]=new_data[col].fillna(new_data[col].mode()[0])
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
total= 'TotalBsmtSF'

sns.scatterplot(x=total, y='SalePrice',data=train,style='Street',markers={'Pave':'^','Grvl':'o'});
sns.scatterplot(x='GrLivArea', y='SalePrice',data=train,color='b',style='Street',markers={'Pave':'^','Grvl':'o'});
var = 'OverallQual'

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=train)

fig.axis(ymin=0, ymax=800000);
f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=train)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);



k = 10 

corrmat = train.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

ax = sns.heatmap(cm, annot=True,annot_kws={'size': 10}, fmt=".2f",xticklabels=cols.values,yticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','SaleType']

sns.pairplot(train[cols], size = 2.5,hue="SaleType", palette="husl")

plt.show();
sns.violinplot(x="SaleType", y="SalePrice", data=train,hue="Street",palette="Set2")
f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='YearBuilt', y="GrLivArea", data=train)

plt.xticks(rotation=90);
train.groupby(['YearBuilt']).SalePrice.aggregate(['mean','std','max']).plot()
train.groupby(['YearBuilt']).GrLivArea.aggregate(['mean','std','max']).plot()
train.sort_values(by = 'GrLivArea', ascending = False)['GrLivArea'][:1]

train = train.drop(train[train.index == 1299].index)
fig = plt.figure()

res = stats.probplot(np.log1p(train['SalePrice']), plot=plt)

train['SalePrice']=np.log1p(train['SalePrice'])
sns.distplot(np.log1p(train['GrLivArea']));

fig = plt.figure()

res = stats.probplot(np.log1p(train['GrLivArea']), plot=plt)

train['GrLivArea']=np.log1p(train['GrLivArea'])
train['HasBsmt']=train['TotalBsmtSF'].apply(lambda x:1 if x!=0 else 0)
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log1p(train['TotalBsmtSF'])
fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)