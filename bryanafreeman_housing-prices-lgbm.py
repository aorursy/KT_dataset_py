import pdb

import pickle

import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import warnings



from scipy import stats

from scipy.special import boxcox1p

from scipy.stats import norm, skew

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer



from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from sklearn.svm import SVR

from sklearn_pandas import DataFrameMapper

from operator import itemgetter

import lightgbm as lgb





import os



%matplotlib inline

warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



# Read the CSV

train_csv_path = '../input/house-prices-advanced-regression-techniques/train.csv'

test_csv_path = '../input/house-prices-advanced-regression-techniques/test.csv'

train_set = pd.read_csv(train_csv_path)

test_set = pd.read_csv(test_csv_path)



# Keep original data clean

train_data = train_set.copy()

test_data = test_set.copy()

train_ids = train_data['Id'].copy()

test_ids = test_data['Id'].copy()

print('Test data original columns: {}'.format(train_data.columns.to_list()))
print('Train data original shape: {}'.format(train_data.shape))

print('Test data original shape: {}'.format(test_data.shape))
train_data.head(5)
train_data['SalePrice'].describe()
plt.figure(figsize=(16, 6))

sns.distplot(train_data['SalePrice']);
fig, ax = plt.subplots(figsize=(23,10))

ax.set(yscale="log")

sns.barplot(x="Neighborhood", y="SalePrice", data=train_data, estimator=np.mean)

plt.show()
var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', figsize=(16, 6), ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 10))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 12))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrs_matrix = train_data.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrs_matrix, vmax=.8, square=True);
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_data[cols], size = 2.5)

plt.show();
total = train_data.isnull().sum().sort_values(ascending=False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('Low range of distribution:')

print(low_range)

print('\nHigh range of the distribution:')

print(high_range)
var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), figsize=(16, 8));
train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]

train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)

train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
sns.distplot(train_data['SalePrice'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train_data['SalePrice'], plot=plt)
train_data['SalePrice'] = np.log(train_data['SalePrice'])

sns.distplot(train_data['SalePrice'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train_data['SalePrice'], plot=plt)
sns.distplot(train_data['GrLivArea'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train_data['GrLivArea'], plot=plt)
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])

sns.distplot(train_data['GrLivArea'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train_data['GrLivArea'], plot=plt)
sns.distplot(train_data['TotalBsmtSF'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
# Revert to clean training data

train_data = train_set.copy()

train_index = train_data.shape[0]

test_index = test_data.shape[0]

target = train_data.SalePrice.values

all_data = pd.concat((train_data, test_data)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data.shape
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['Alley'] = all_data['Alley'].fillna('None')

all_data['Fence'] = all_data['Fence'].fillna('None')

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('FireplaceQu')

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for col in columns:

    encoder = LabelEncoder() 

    encoder.fit(list(all_data[col].values)) 

    all_data[col] = encoder.transform(list(all_data[col].values))

    

all_data.shape
sale_year = np.max(all_data['YrSold'])

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data.shape
num_features = all_data.dtypes[all_data.dtypes != 'object'].index

skewed_features = all_data[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print('{} skewed features to transform'.format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)

all_data.shape
train_data = all_data[:train_index]

test_data = all_data[train_index:]
X_train, X_val, y_train, y_val = train_test_split(

    train_data, target, test_size=0.20, random_state=42)
model = lgb.LGBMRegressor(objective='regression',

                         num_leaves=5,

                         learning_rate=0.05,

                         n_estimators=720,

                         max_bin = 55,

                         bagging_fraction = 0.8,

                         bagging_freq = 5,

                         feature_fraction = 0.2319,

                         feature_fraction_seed=9,

                         bagging_seed=9,

                         min_data_in_leaf =6,

                         min_sum_hessian_in_leaf = 11)
n_folds = 5



y_train_scaled = np.log1p(y_train)

y_val_scaled = np.log1p(y_val)



def rmse_cv(model, X, y):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))

    return rmse
X = X_train

y = y_train_scaled



score = rmse_cv(model, X, y)

print("Model score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
def rmse_cv_val(model, X, y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))

    return rmse
X = X_val

y = y_val_scaled





score = rmse_cv_val(model, X, y)

print("Validation score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
X = X_train

y = y_train_scaled

model.fit(X, y)
X = test_data

predictions = np.exp(model.predict(X))

result=pd.DataFrame({'Id':test_ids, 'SalePrice':predictions})

result.to_csv('/kaggle/working/submission.csv',index=False)

print('done')