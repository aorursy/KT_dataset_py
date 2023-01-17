# Importing relevant libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time

import datetime
# Importing the data for setting up the analysis



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
train.head()

train.info()

train.shape
# Exploring the distribution of SalePrice

plt.style.use('ggplot')

plt.figure(figsize=(12,7))

sns.distplot(train.SalePrice, bins=20)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,1))

plt.ylabel("Number of Houses")

plt.xlabel("Price of Houses")

plt.show()
train.SalePrice.skew()
#Log transforming the SalePrice for making the distribution more Normal

train.SalePrice = np.log(train.SalePrice)

train.SalePrice.skew()
# Exploring the distribution of SalePrice after log transformation

plt.style.use('ggplot')

plt.figure(figsize=(12,7))

sns.distplot(train.SalePrice, bins=20)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,1))

plt.ylabel("Number of Houses")

plt.xlabel("Price of Houses")

plt.show()
# Understanding the Split of Different Data Types of the Feature Variables

train.dtypes.value_counts()
num_features = train.select_dtypes(include=[np.number])

num_features.dtypes

num_list = num_features.columns

print(num_list)
corr = num_features.corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(corr, square=True)
corr1 = num_features[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']].corr()

sns.heatmap(corr1, annot=True, annot_kws={'size':8})
sns.pairplot(num_features[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']])

plt.show()
# Distribution of Year and effect of YearBuilt on SalePrice

plt.figure(figsize=(20,15))

sns.jointplot('YearBuilt','SalePrice', data=num_features, alpha=0.5, color='purple')

plt.show()
# Plot of Garage built year with House Built Year

plt.figure(figsize=(12,7))

plt.plot(num_features.YearBuilt, num_features.GarageYrBlt, '.', alpha=0.4)

plt.xlabel('House Year Built')

plt.ylabel('Garage Year Built')

plt.show()
sns.lmplot('MasVnrArea', 'SalePrice', data=num_features, palette='muted', scatter_kws={"s":30})

plt.xlabel("Masonry Veneer Area")

plt.show()
cat_features = train.select_dtypes(include=[object])

cat_features.dtypes
# Values of different categorical feqatures and their counts

melted = cat_features.melt(var_name='Cat_Features', value_name='Feature Values')

melted.groupby(['Cat_Features', 'Feature Values']).size()
plt.figure(figsize=(15,7))

sns.swarmplot('Neighborhood', 'SalePrice', data=train, order = np.sort(train.Neighborhood.unique()))

plt.xticks(rotation=45)

plt.show()
cat_features.Neighborhood.value_counts()
plt.figure(figsize=(15,7))

sns.stripplot('Heating', 'SalePrice', data=train, order = np.sort(train.Heating.unique()))

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,7))

sns.boxplot('GarageType', 'SalePrice', data=train)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,7))

sns.boxplot('FireplaceQu', 'SalePrice', data=train)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,7))

sns.violinplot('BsmtQual', 'SalePrice', data=train)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,7))

sns.violinplot('Exterior1st', 'SalePrice', data=train)

plt.xticks(rotation=45)

plt.show()
train['Source'] = 'train'

test['Source'] = 'test'

combined = pd.concat([train, test], ignore_index=True, sort=False)

print(train.shape, test.shape, combined.shape)
# Normalizing outliers in GarageArea

GarageArea_mean = combined.GarageArea.mean()

func = lambda x: x.GarageArea > 1250 and GarageArea_mean or x.GarageArea

combined.GarageArea = combined.apply(func, axis=1).astype(float)
# Normalizing outlier in GrLivArea

GrLivArea_mean = combined.GrLivArea.mean()

func = lambda x: x.GrLivArea > 4000 and GrLivArea_mean or x.GrLivArea

combined.GrLivArea = combined.apply(func, axis=1).astype(float)
print('Original missing value:', format(combined.isnull().sum().sum()))
# % of Null values in each feature variable

null_values = combined.isnull().sum() / combined.shape[0] * 100

# features having more than 

null_values[null_values > 50]
combined.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
null_values = combined.isnull().sum() / combined.shape[0] * 100

#features with 0 to 50% missing values

null50 = null_values[(null_values > 0) & (null_values < 50)]
#filtering only categorical features in null50

[x for x in null50.index if x not in num_list]
cat_feat = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'SaleType']

combined.update(combined[cat_feat].fillna(combined[cat_feat].mode().iloc[0]))
# filtering only numerical features in null50

[x for x in null50.index if x in num_list]
combined.update(combined[['LotFrontage',

 'MasVnrArea',

 'BsmtFinSF1',

 'BsmtFinSF2',

 'BsmtUnfSF',

 'TotalBsmtSF',

 'BsmtFullBath',

 'BsmtHalfBath',

 'GarageCars',

 'GarageArea',]].fillna(0))
combined.update(combined['GarageYrBlt'].fillna(combined.YearBuilt))
print('Final missing value:', format(combined.isnull().sum().sum()))
cat_features.columns
combined.LotShape.replace({'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}, inplace=True)

combined.MSZoning.replace({'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}, inplace=True)

combined.ExterQual.replace({'Gd':2, 'TA':1, 'Ex':2, 'Fa':0}, inplace=True)

combined.ExterCond.replace({'TA':2, 'Gd':3, 'Fa':1, 'Po':0, 'Ex':4}, inplace=True)

combined.BsmtQual.replace({'Gd':2, 'TA':1, 'Ex':3, 'Fa':0}, inplace=True)

combined.BsmtCond.replace({'Gd':3, 'TA':2, 'Po':0, 'Fa':1}, inplace=True)

combined.BsmtExposure.replace({'No':0, 'Gd':3, 'Mn':1, 'Av':2}, inplace=True)

combined.BsmtFinType1.replace({'GLQ':5, 'ALQ':4, 'Unf':0, 'Rec':2, 'BLQ':3, 'LwQ':1}, inplace=True)

combined.BsmtFinType2.replace({'GLQ':5, 'ALQ':4, 'Unf':0, 'Rec':2, 'BLQ':3, 'LwQ':1}, inplace=True)

combined.HeatingQC.replace({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}, inplace=True)

combined.CentralAir.replace({'Y':1, 'N':0}, inplace=True)

combined.KitchenQual.replace({'Gd':2, 'TA':1, 'Ex':3, 'Fa':0}, inplace=True)

combined.Functional.replace({'Typ':6, 'Min1':5, 'Maj1':2, 'Min2':4, 'Mod':3, 'Maj2':1, 'Sev':0}, inplace=True)

combined.FireplaceQu.replace({'Gd':3, 'TA':2, 'Fa':1, 'Ex':4, 'Po':0}, inplace=True)

combined.GarageFinish.replace({'RFn':1, 'Unf':0, 'Fin':2}, inplace=True)

combined.GarageQual.replace({'TA':2, 'Fa':1, 'Gd':3, 'Ex':4, 'Po':0}, inplace=True)

combined.GarageCond.replace({'TA':2, 'Fa':1, 'Gd':3, 'Ex':4, 'Po':0}, inplace=True)

combined.PavedDrive.replace({'Y':2, 'N':0, 'P':1}, inplace=True)

#One Hot Encoding of Categorical Features which are nominal

cat1 = ['MSZoning', 'RoofMatl', 'Street', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood','Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']

dfcat_onehot = combined[cat1]

cat_onehot = pd.get_dummies(dfcat_onehot, drop_first=True)
combined.drop(cat1, axis=1, inplace=True)
combined = pd.concat([combined, cat_onehot], axis=1)
combined.drop(['GarageArea', 'TotRmsAbvGrd', '1stFlrSF', 'GarageYrBlt'], axis=1, inplace=True)
#drop the ID column

combined.drop('Id', axis=1, inplace=True)
combined.dtypes.value_counts()
train1 = combined.loc[combined.Source == 'train']

test1 = combined.loc[combined.Source == 'test']
target = train1['SalePrice']
train1.drop(['Source', 'SalePrice'], axis=1, inplace=True)

test1.drop(['Source', 'SalePrice'], axis=1, inplace=True)
train_df = train1

test_df = test1
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

X = train_df

y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(reg.score(X_test, y_test))

print(mean_squared_error(y_test, y_pred))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



reg1 = LinearRegression()

reg1.fit(train_df, target)

y_pred1 = reg1.predict(test_df)

lreg_ex = np.exp(y_pred1)

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



param_grid = {'alpha': np.linspace(0, 1, 2000)}

feats = train_df.columns

lasso = Lasso(max_iter=100000)

gm_cv = GridSearchCV(lasso, param_grid, cv=5)

gm_cv.fit(train_df, target)

y_pred2 = gm_cv.predict(test_df)

print(gm_cv.best_params_)

print(gm_cv.best_score_)

lasso_exp = np.exp(y_pred2)

from sklearn.linear_model import ElasticNet



l1_space = np.linspace(0, 1, 1000)

param_grid2 = {'l1_ratio': l1_space}

elastic_net = ElasticNet()

gm_cv2 = GridSearchCV(elastic_net, param_grid2, cv=5)

gm_cv2.fit(train_df, target)

y_pred3 = gm_cv2.predict(test_df)

print(gm_cv2.best_params_)

print(gm_cv2.best_score_)

elastic_exp = np.exp(y_pred3)
