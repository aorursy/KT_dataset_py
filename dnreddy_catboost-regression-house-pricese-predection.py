import pandas as pd

import numpy as np



from matplotlib import cm

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',header=0)

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', header = 0)

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv',header = 0)
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
train.head()
test.head()
sample_submission.head()
print(train.shape)

print(test.shape)

print(sample_submission.shape)
train['SalePrice'].describe()
#histogram

sns.distplot(train['SalePrice']);
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
sns.set(font_scale=1)

correlation_train=train.corr()

plt.figure(figsize=(30,20))

sns.heatmap(correlation_train,annot=True,fmt='.1f')
train.corr()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
#missing data

train_total = train.isnull().sum().sort_values(ascending=False)

train_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

train_missing_data = pd.concat([train_total, train_percent], axis=1, keys=['Total', 'Percent'])

train_missing_data.head(20)
#missing data in test

test_total = test.isnull().sum().sort_values(ascending=False)

test_percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

test_missing_data = pd.concat([test_total, test_percent], axis=1, keys=['Total', 'Percent'])

test_missing_data.head(35)
# Train Data Imputation



for col in ('PoolQC','MiscFeature','Alley','Fence','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass','FireplaceQu'):

    train[col] = train[col].fillna('None')

    



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',

            'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    train[col] = train[col].fillna(0)

    



train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



train["Functional"] = train["Functional"].fillna("Typ")



train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])

train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])

train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])

train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])





train = train.drop(['Utilities'], axis=1)
train_total = train.isnull().sum().sort_values(ascending=False)

train_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

train_missing_data = pd.concat([train_total, train_percent], axis=1, keys=['Total', 'Percent'])

train_missing_data.head(5)
# Test Data Imputation



for col in ('PoolQC','MiscFeature','Alley','Fence','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass','FireplaceQu'):

    test[col] = test[col].fillna('None')

    



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',

            'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    test[col] = test[col].fillna(0)

    



test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



test["Functional"] = test["Functional"].fillna("Typ")



test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])





test = test.drop(['Utilities'], axis=1)
#missing data in test

test_total = test.isnull().sum().sort_values(ascending=False)

test_percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

test_missing_data = pd.concat([test_total, test_percent], axis=1, keys=['Total', 'Percent'])

test_missing_data.head(5)
train.head(5)
test.head(5)
# Print the number of unique levels in train and test data

train_unique = train.nunique().sort_values(ascending=False)

test_unique = test.nunique().sort_values(ascending=False)

unique_data = pd.concat([train_unique, test_unique], axis=1, keys=['Train', 'Test'],join="inner")

unique_data.head(100)
# Train data

train['LotFrontage'] = train['LotFrontage'].astype('int64')

train['MasVnrArea'] = train['MasVnrArea'].astype('int64')

train['GarageYrBlt'] = train['GarageYrBlt'].astype('int64')

# Test Data

test['LotFrontage'] = test['LotFrontage'].astype('int64')

test['MasVnrArea'] = test['MasVnrArea'].astype('int64')

test['GarageYrBlt'] = test['GarageYrBlt'].astype('int64')

test['BsmtFinSF1'] = test['BsmtFinSF1'].astype('int64')

test['BsmtFinSF2'] = test['BsmtFinSF2'].astype('int64')

test['BsmtUnfSF'] = test['BsmtUnfSF'].astype('int64')

test['TotalBsmtSF'] = test['TotalBsmtSF'].astype('int64')

test['GarageArea'] = test['GarageArea'].astype('int64')

test['BsmtFullBath'] = test['BsmtFullBath'].astype('int64')

test['BsmtHalfBath'] = test['BsmtHalfBath'].astype('int64')
for col in ('Alley','BldgType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual',

            'CentralAir','Condition1','ExterCond','ExterQual','Fence','FireplaceQu','Foundation',

            'Functional','GarageCond','GarageFinish','GarageType','HeatingQC','KitchenQual',

            'LandContour','LandSlope','LotConfig','LotShape','MasVnrType','MSZoning',

            'Neighborhood','PavedDrive','RoofStyle','SaleCondition','SaleType','Street',

           'OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','HalfBath','TotRmsAbvGrd',

           'MoSold','YrSold') :

    train[col] = train[col].astype('category')

    test[col] = test[col].astype('category')

# Train data

train.drop(['Condition2','Electrical','Exterior1st','Exterior2nd','GarageQual',

            'Heating','HouseStyle','MiscFeature','PoolQC','RoofMatl',

           'MSSubClass','LowQualFinSF','FullBath','BedroomAbvGr','KitchenAbvGr',

            'Fireplaces','GarageCars','3SsnPorch','PoolArea','MiscVal','Id'], axis=1, inplace=True)

# Test data

test.drop(['Condition2','Electrical','Exterior1st','Exterior2nd','GarageQual',

            'Heating','HouseStyle','MiscFeature','PoolQC','RoofMatl',

           'MSSubClass','LowQualFinSF','FullBath','BedroomAbvGr','KitchenAbvGr',

            'Fireplaces','GarageCars','3SsnPorch','PoolArea','MiscVal','Id'], axis=1, inplace=True)
train.shape
test.shape
train.head()
test.head()
categorical_features_indices = np.where(train.dtypes != np.int64)[0]
categorical_features_indices
from sklearn.model_selection import train_test_split



y = train["SalePrice"]

X = train.drop('SalePrice', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=789)
from catboost import CatBoostRegressor
model_1=CatBoostRegressor(loss_function='RMSE')

model_1.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
pred =model_1.predict(test)
submission = pd.DataFrame()

submission['Id'] = sample_submission.Id

submission['SalePrice'] = pred

submission.to_csv('submission.csv', index=False)

submission.head(5)