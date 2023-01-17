# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Ridge

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore  warning (from sklearn)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
house_train = pd.read_csv('../input/train.csv')

house_test = pd.read_csv('../input/test.csv')
#drop meaningless feature 'Id' 

house_train = house_train.drop(['Id'],axis=1)

house_test = house_test.drop(['Id'],axis=1)
#drop the features with more 30% data missing

threshold = 0.3*len(house_train)

df = pd.DataFrame(len(house_train)-house_train.count(),columns=['empty'])

df.index[df['empty']>threshold]
house_train = house_train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],axis=1)

house_test = house_test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],axis=1)
house_train.columns
#using the heatmap to visulize correlated features

corr_matrix = house_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_matrix, vmax=.8, square=True);
#sort the coeffient matrix by 'SalePrice'

corr_matrix=corr_matrix.sort_values(by=['SalePrice'],axis=0,ascending=False)

corr_matrix=corr_matrix.sort_values(by=['SalePrice'],axis=1,ascending=False)
sns.set(font_scale=1.2)

f, ax = plt.subplots(figsize=(21, 18))

heat_map = sns.heatmap(corr_matrix,cbar=True,annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=corr_matrix.columns, xticklabels=corr_matrix.columns)

plt.show()
corr_matrix.columns
# drop less correlated coefficents

house_train = house_train.drop(['ScreenPorch',

       'PoolArea', 'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath',

       'MiscVal', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass',

       'EnclosedPorch', 'KitchenAbvGr'],axis=1)

house_test = house_test.drop(['ScreenPorch',

       'PoolArea', 'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath',

       'MiscVal', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass',

       'EnclosedPorch', 'KitchenAbvGr'],axis=1)
#There is linear relationship between GarageCars and GarageArea

#FullBath, HalfBath, BasementFullBath are pair problems

#Total room above grade is propotional to the sum of 1st and 2nd square foot
house_train = house_train.drop(['GarageArea','HalfBath','BsmtFullBath','TotRmsAbvGrd'],axis=1)

house_test = house_test.drop(['GarageArea','HalfBath','BsmtFullBath','TotRmsAbvGrd'],axis=1)
house_train.columns
# find the rest missing data

all_data = pd.concat((house_train.drop(["SalePrice"], axis=1), house_test))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

plt.figure(figsize=(12, 6))

plt.xticks(rotation="90")

sns.barplot(x=all_data_na.index, y=all_data_na)
# LotFrontage

house_train['LotFrontage'] = house_train['LotFrontage'].fillna(method='ffill')

house_test['LotFrontage'] = house_test['LotFrontage'].fillna(method='ffill')



#Garage

house_train['GarageQual'] = house_train['GarageQual'].fillna('None')

house_test['GarageQual'] = house_test['GarageQual'].fillna('None')

house_train['GarageFinish'] = house_train['GarageFinish'].fillna('None')

house_test['GarageFinish'] = house_test['GarageFinish'].fillna('None')

house_train = house_train.drop(['GarageYrBlt'],axis=1)

house_test = house_test.drop(['GarageYrBlt'],axis=1)

house_train['GarageCond'] = house_train['GarageCond'].fillna('None')

house_test['GarageCond'] = house_test['GarageCond'].fillna('None')

house_train['GarageType'] = house_train['GarageType'].fillna('None')

house_test['GarageType'] = house_test['GarageType'].fillna('None')

house_train['GarageCars'] = house_train['GarageCars'].fillna(0)

house_test['GarageCars'] = house_test['GarageCars'].fillna(0)



#Basement

house_train['BsmtExposure'] = house_train['BsmtExposure'].fillna('None')

house_test['BsmtExposure'] = house_test['BsmtExposure'].fillna('None')

house_train['BsmtCond'] = house_train['BsmtCond'].fillna('None')

house_test['BsmtCond'] = house_test['BsmtCond'].fillna('None')

house_train['BsmtQual'] = house_train['BsmtQual'].fillna('None')

house_test['BsmtQual'] = house_test['BsmtQual'].fillna('None')

house_train['BsmtFinType1'] = house_train['BsmtFinType1'].fillna('None')

house_test['BsmtFinType1'] = house_test['BsmtFinType1'].fillna('None')

house_train['BsmtFinType2'] = house_train['BsmtFinType2'].fillna('None')

house_test['BsmtFinType2'] = house_test['BsmtFinType2'].fillna('None')

house_train = house_train.drop(['BsmtFinSF1','BsmtUnfSF'],axis=1)

house_test = house_test.drop(['BsmtFinSF1','BsmtUnfSF'],axis=1)

house_train['TotalBsmtSF'] = house_train['TotalBsmtSF'].fillna(0)

house_test['TotalBsmtSF'] = house_test['TotalBsmtSF'].fillna(0)



#Masonry veneer

house_train['MasVnrType'] = house_train['MasVnrType'].fillna('None')

house_test['MasVnrType'] = house_test['MasVnrType'].fillna('None')

house_train['MasVnrArea'] = house_train['MasVnrArea'].fillna(0)

house_test['MasVnrArea'] = house_test['MasVnrArea'].fillna(0)



#MSZoning

house_train['MSZoning'] = house_train['MSZoning'].fillna('None')

house_test['MSZoning'] = house_test['MSZoning'].fillna('None')



#Functional

house_train['Functional'] = house_train['Functional'].fillna('Typ')

house_test['Functional'] = house_test['Functional'].fillna('Typ')



#Utillities

house_train['Utilities'] = house_train['Utilities'].fillna('NoSeWa')

house_test['Utilities'] = house_test['Utilities'].fillna('NoSeWa')



#Electrical FuseA  

house_train['Electrical'] = house_train['Electrical'].fillna('FuseA')

house_test['Electrical'] = house_test['Electrical'].fillna('FuseA')



#Kitchen

house_train['KitchenQual'] = house_train['KitchenQual'].fillna('TA')

house_test['KitchenQual'] = house_test['KitchenQual'].fillna('TA')



#Exterior

house_train['Exterior1st'] = house_train['Exterior1st'].fillna('Other')

house_test['Exterior1st'] = house_test['Exterior1st'].fillna('Other')

house_train['Exterior2nd'] = house_train['Exterior2nd'].fillna('Other')

house_test['Exterior2nd'] = house_test['Exterior2nd'].fillna('Other')



#SaleType

house_train['SaleType'] = house_train['SaleType'].fillna('Oth')

house_test['SaleType'] = house_test['SaleType'].fillna('Oth')
train_dummies = pd.get_dummies(pd.concat((house_train.drop(["SalePrice"], axis=1), house_test))).iloc[: house_train.shape[0]]

test_dummies = pd.get_dummies(pd.concat((house_train.drop(["SalePrice"], axis=1), house_test))).iloc[house_train.shape[0]:]
y = house_train['SalePrice']

y = np.log(y+1)
#Using Ridge to explore outliers

out_explore = Ridge(alpha=10)

out_explore.fit(train_dummies, y)

np.sqrt(-cross_val_score(out_explore, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
prediction_y = out_explore.predict(train_dummies)

err = prediction_y - y

Mean_err = err.mean()

Std_err = err.std()

z = (err - Mean_err) / Std_err

z = np.array(z)

outliers1 = np.where(abs(z) > abs(z).std() * 3)[0]

outliers1
#Delete Outliers

house_train = house_train.drop(outliers1)

train_dummies = pd.get_dummies(pd.concat((house_train.drop(["SalePrice"], axis=1), house_test))).iloc[: house_train.shape[0]]

test_dummies = pd.get_dummies(pd.concat((house_train.drop(["SalePrice"], axis=1), house_test))).iloc[house_train.shape[0]:]

y = house_train['SalePrice']

y = np.log(y+1)
gbr = GradientBoostingRegressor(max_depth=6, n_estimators=150)

gbr.fit(train_dummies, y)

np.sqrt(-cross_val_score(gbr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
gbr.score(train_dummies,y)
pre_y = gbr.predict(train_dummies)
plt.figure(figsize=(6, 6))

plt.scatter(y, pre_y)

plt.plot(range(10, 15), range(10, 15), color="red")
pre_test = gbr.predict(test_dummies)

pre_df = pd.DataFrame(pre_test)

pre_df["SalePrice"] = pre_test

pre_df = pre_df[["SalePrice"]]

pre_test = np.array(pre_df.SalePrice)

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['SalePrice'] = np.exp(pre_test)-1

sample_submission.to_csv("my_submission.csv", index=False)
