#Importing all the required libraries

import pandas as pd

import seaborn as sns

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
# Reading the train and test dataset

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#Printing out the number of samples and features of Training as well as test dataset



print("{} no of features with {} numbers of samples in training".format(train.shape[1],train.shape[0]))

print("{} no of features with {} numbers of samples in testing".format(test.shape[1],test.shape[0]))
train_id = train['Id']

test_id = test['Id']

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)

print("{} no of features with {} numbers of samples in training".format(train.shape[1],train.shape[0]))

print("{} no of features with {} numbers of samples in testing".format(test.shape[1],test.shape[0]))
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

full_data = pd.concat((train, test)).reset_index(drop=True)

full_data.drop(['SalePrice'], axis=1, inplace=True)

print("full_data size is : {}".format(full_data.shape))
full_data_null = (full_data.isnull().sum() / len(full_data)) * 100

full_data_null = full_data_null.drop(full_data_null[full_data_null == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Data Percentage' :full_data_null})

missing_data
full_data["PoolQC"] = full_data["PoolQC"].fillna("None")

full_data["MiscFeature"] = full_data["MiscFeature"].fillna("None")

full_data["Alley"] = full_data["Alley"].fillna("None")

full_data["Fence"] = full_data["Fence"].fillna("None")

full_data["FireplaceQu"] = full_data["FireplaceQu"].fillna("None")

full_data["MasVnrType"] = full_data["MasVnrType"].fillna("None")

full_data["MasVnrArea"] = full_data["MasVnrArea"].fillna(0)

full_data['MSSubClass'] = full_data['MSSubClass'].fillna("None")

full_data = full_data.drop(['Utilities'], axis=1)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    full_data[col] = full_data[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    full_data[col] = full_data[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    full_data[col] = full_data[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    full_data[col] = full_data[col].fillna('None')
full_data['MSZoning'] = full_data.groupby("Neighborhood")["MSZoning"].transform(lambda x: x.fillna(x.mode()))

full_data["Functional"] = full_data["Functional"].fillna("Typ")

full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

full_data['Electrical'] = full_data.groupby("Neighborhood")["Electrical"].transform(lambda x: x.fillna(x.mode()[0]))

full_data['KitchenQual'] = full_data.groupby("Neighborhood")["KitchenQual"].transform(lambda x: x.fillna(x.mode()[0]))

full_data['Exterior1st'] = full_data.groupby("Neighborhood")["Exterior1st"].transform(lambda x: x.fillna(x.mode()[0]))

full_data['Exterior2nd'] = full_data.groupby("Neighborhood")["Exterior2nd"].transform(lambda x: x.fillna(x.mode()[0]))

full_data['SaleType'] = full_data.groupby("Neighborhood")["SaleType"].transform(lambda x: x.fillna(x.mode()[0]))
#Check remaining missing values if any 

full_data_null = (full_data.isnull().sum() / len(full_data)) * 100

missing_data = pd.DataFrame({'Missing Percentage' :full_data_null})

missing_data.head()
corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

corr = corr.SalePrice

display(corr)
train.head()
plot = plt.subplot()

plot.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
plot = plt.subplot()

plot.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
plot = plt.subplot()

plot.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)