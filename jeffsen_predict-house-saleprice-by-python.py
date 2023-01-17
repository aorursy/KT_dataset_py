import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import LinearRegression

% matplotlib inline
#导入数据集

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
#查看字段

train_df.columns
#查看各字段的信息

train_df.info()

print('_'*40)

test_df.info()
#可以看到Alley、FireplaceQu、PoolQC、Fence、MiscFeature等字段有太多的缺损值了，对于构建模型不会有太多的帮助，因此drop掉这些数据

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

#剔除无效字段

train_df = train_df.drop(['Alley', 'PoolQC','Fence','MiscFeature','FireplaceQu','Utilities', 'Street','LandSlope'], axis=1)

test_df = test_df.drop(['Alley', 'PoolQC','Fence','MiscFeature','FireplaceQu','Utilities', 'Street','LandSlope'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
train_df.describe(include=['O'])
#MSZoning与saleprice的关系

train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#LandContour与saleprice的关系

train_df[['LandContour', 'SalePrice']].groupby(['LandContour'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#HouseStyle与saleprice的关系

train_df[['HouseStyle', 'SalePrice']].groupby(['HouseStyle'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#RoofStyle与saleprice的关系

train_df[['RoofStyle', 'SalePrice']].groupby(['RoofStyle'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#ExterQual与saleprice的关系

train_df[['ExterQual', 'SalePrice']].groupby(['ExterQual'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#Heating与saleprice的关系

train_df[['Heating', 'SalePrice']].groupby(['Heating'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#CentralAir与saleprice的关系

train_df[['CentralAir', 'SalePrice']].groupby(['CentralAir'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#KitchenQual与Saleprice的关系

train_df[['KitchenQual', 'SalePrice']].groupby(['KitchenQual'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#SaleCondition与saleprice的关系

train_df[['SaleCondition', 'SalePrice']].groupby(['SaleCondition'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
#box plot overallqual/saleprice

train_df['KitchenQual']=train_df['KitchenQual'].map({'TA':0,'Gd':1,'Ex':2,'Fa':3}).astype(int)

test_df['KitchenQual']=test_df['KitchenQual'].map({'TA':0,'Gd':1,'Ex':2,'Fa':3,np.NaN:4}).astype(int)
train_df['MSZoning']=train_df['MSZoning'].map({'FV':0,'RL':1,'RH':2,'RM':3,'C (all)':4}).astype(int)



test_df['MSZoning']=test_df['MSZoning'].map({'FV':0,'RL':1,'RH':2,'RM':3,'C (all)':4,np.NaN:5}).astype(int)
for dataset in combine:

    dataset['LandContour'] = dataset['LandContour'].map( {'HLS': 0, 'Low': 1, 'Lvl': 2,'Bnk':3} ).astype(int)

    dataset['HouseStyle'] = dataset['HouseStyle'].map( {'2.5Fin': 0, '2Story': 1, '1Story': 2,'SLvl':3,'2.5Unf':4,'1.5Fin':5,'SFoyer':6,'1.5Unf':7} ).astype(int)

    dataset['RoofStyle'] = dataset['RoofStyle'].map( {'Shed': 0, 'Hip': 1, 'Flat': 2,'Mansard':3,'Gable':4,'Gambrel':5} ).astype(int)

    dataset['ExterQual'] = dataset['ExterQual'].map( {'Ex': 0, 'Gd': 1, 'TA': 2,'Fa':3} ).astype(int)

    dataset['Heating'] = dataset['Heating'].map( {'GasA': 0, 'GasW': 1, 'OthW': 2,'Wall':3,'Grav':4,'Floor':5} ).astype(int)

    dataset['CentralAir'] = dataset['CentralAir'].map( {'Y': 0, 'N': 1} ).astype(int)

    dataset['SaleCondition'] = dataset['SaleCondition'].map( {'Partial': 0, 'Normal': 1, 'Alloca': 2,'Family':3,'Abnorml':4,'AdjLand':5} ).astype(int)

#查看saleprice的分布情况

sns.distplot(train_df['SalePrice']);
#correlation matrix

var = 'MSSubClass'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#TotalBsmtSF Vs SalePrice 呈现线性关系

var = 'TotalBsmtSF'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
# YearBuilt vs saleprice

var = 'YearBuilt'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 4))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#FullBath vs saleprice

var = 'FullBath'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual',  'TotalBsmtSF',  'YearBuilt','MSSubClass']

sns.pairplot(train_df[cols], size = 1.5)

plt.show();
#在测试集中'TotalBsmtSF'含有na值，填充成0

test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna('0')
#missing data

train=train_df[['KitchenQual','MSZoning','LandContour','HouseStyle','RoofStyle','ExterQual',

                'Heating','CentralAir','SaleCondition','MSSubClass','TotalBsmtSF','Fireplaces','YearBuilt','OverallQual','SalePrice']]

test=test_df[['KitchenQual','MSZoning','LandContour','HouseStyle','RoofStyle','ExterQual',

                'Heating','CentralAir','SaleCondition','MSSubClass','TotalBsmtSF','Fireplaces','YearBuilt','OverallQual']]
#查看处理后的训练集和测试集

train.head()

test.head()
#查看测试集与训练集的shape

X_train = train.drop('SalePrice', axis=1)

Y_train = train['SalePrice']

X_test  = test

X_train.shape, Y_train.shape, X_test.shape
#通过线性回归建模

linReg = LinearRegression()

linReg.fit(X_train, Y_train)
print(X_test.info())



linReg.score(X_train, Y_train)

#随机森林预测

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest