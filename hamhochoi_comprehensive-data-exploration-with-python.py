import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#bring in the six packs
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_train_copy = df_train.copy()
df_test_copy  = df_test.copy()
#check the decoration
df_train_copy.columns
df_train_copy.describe()
df_train_copy.info()
#descriptive statistics summary
df_train_copy['SalePrice'].describe()

#histogram
sns.distplot(df_train_copy['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train_copy['SalePrice'].skew())
print("Kurtosis: %f" % df_train_copy['SalePrice'].kurt())
#correlation matrix
corrmat = df_train_copy.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train_copy[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train_copy[cols], size = 2.5)
plt.show();
# Get columns have type is object == Category 
category_column_index = np.argwhere(df_train_copy.dtypes=="object")
category_columns = df_train_copy.columns[category_column_index]
category_columns = [i[0] for i in category_columns]
category_columns
df_train_copy[category_columns] = df_train_copy[category_columns].astype('category')
cat_columns = df_train_copy.select_dtypes(['category']).columns
cat_columns
df_train_copy[cat_columns] = df_train_copy[cat_columns].apply(lambda x: x.cat.codes)
df_train_copy.head(5)
df_train_copy.info()
#correlation matrix
corrmat = df_train_copy.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 25 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train_copy[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7.5}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt','1stFlrSF', 'TotalBsmtSF', 'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'Foundation']
sns.pairplot(df_train_copy[cols], size = 2.5)
plt.show();
fig, ax = plt.subplots()
ax.scatter(x = df_train_copy['GrLivArea'], y = df_train_copy['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# Deleting outliers
# Just delete in df_train_copy
df_train_copy = df_train_copy.drop(df_train_copy[(df_train_copy['GrLivArea']>4000) & (df_train_copy['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df_train_copy['GrLivArea'], df_train_copy['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#histogram and normal probability plot
sns.distplot(df_train_copy['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train_copy['SalePrice'], plot=plt)
df_train_copy['SalePrice'] = np.log(df_train_copy['SalePrice'] + 1)

#transformed histogram and normal probability plot
sns.distplot(df_train_copy['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train_copy['SalePrice'], plot=plt)
all_data = pd.concat((df_train_copy, df_test_copy))
all_data_copy = all_data.copy()
# Get columns have type is object == Category 
category_column_index = np.argwhere(all_data_copy.dtypes=="object")
category_columns = all_data_copy.columns[category_column_index]
category_columns = [i[0] for i in category_columns]
category_columns
all_data_copy[category_columns] = all_data_copy[category_columns].astype('category')
cat_columns = all_data_copy.select_dtypes(['category']).columns
cat_columns
all_data_copy[cat_columns] = all_data_copy[cat_columns].apply(lambda x: x.cat.codes)
#missing data
total = all_data_copy.isnull().sum().sort_values(ascending=False)
percent = (all_data_copy.isnull().sum()/all_data_copy.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['No. Missing', 'Percent'])
missing_data.head(20)
# Fill NA value with mean value of the variables
all_data_copy["LotFrontage"] = all_data_copy["LotFrontage"].fillna(all_data_copy["LotFrontage"].median())
all_data_copy["GarageYrBlt"] = all_data_copy["LotFrontage"].fillna(all_data_copy["LotFrontage"].median())
all_data_copy["MasVnrArea"] = all_data_copy["LotFrontage"].fillna(all_data_copy["LotFrontage"].median())
all_data_copy["BsmtHalfBath"] = all_data_copy["BsmtHalfBath"].fillna(all_data_copy["BsmtHalfBath"].median())
all_data_copy["BsmtFullBath"] = all_data_copy["BsmtFullBath"].fillna(all_data_copy["BsmtFullBath"].median())
all_data_copy["BsmtFinSF2"] = all_data_copy["BsmtFinSF2"].fillna(all_data_copy["BsmtFinSF2"].median())
all_data_copy["TotalBsmtSF"] = all_data_copy["TotalBsmtSF"].fillna(all_data_copy["TotalBsmtSF"].median())
all_data_copy["GarageCars"] = all_data_copy["GarageCars"].fillna(all_data_copy["GarageCars"].median())
all_data_copy["GarageArea"] = all_data_copy["GarageArea"].fillna(all_data_copy["GarageArea"].median())
all_data_copy["BsmtFinSF1"] = all_data_copy["BsmtFinSF1"].fillna(all_data_copy["BsmtFinSF1"].median())
all_data_copy["BsmtUnfSF"] = all_data_copy["BsmtUnfSF"].fillna(all_data_copy["BsmtUnfSF"].median())

#missing data
total = all_data_copy.isnull().sum().sort_values(ascending=False)
percent = (all_data_copy.isnull().sum()/all_data_copy.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['No. Missing', 'Percent'])
missing_data.head(20)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


from sklearn.linear_model import LinearRegression
all_data_copy.info()
# X_train = all_data_copy.drop(['SalePrice', 'Id'], axis=1).head(len(df_train_copy)) # -2 because of delete 2 outlier
# Y_train = all_data_copy['SalePrice'].head(len(df_train_copy))

# X_test = all_data_copy[len(df_train_copy)::].drop(['SalePrice', 'Id'], axis=1)

# 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt','1stFlrSF', 'TotalBsmtSF', 'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'Foundation', 'BsmtFinSF1', 'LotFrontage', 'GarageCond', 'GarageQual', 'CentralAir', 'WoodDeckSF', 'OpenPorchSF', '2ndFlrSF', 'HalfBath'
X_train = all_data_copy[['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt','1stFlrSF', 'TotalBsmtSF', 'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'Foundation', 'GarageCond', 'GarageQual', 'CentralAir', 'WoodDeckSF', 'OpenPorchSF', '2ndFlrSF', 'HalfBath'
]].head(len(df_train_copy))
Y_train = all_data_copy['SalePrice'].head(len(df_train_copy))
X_test = all_data_copy[['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt','1stFlrSF', 'TotalBsmtSF', 'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'Foundation', 'GarageCond', 'GarageQual', 'CentralAir', 'WoodDeckSF', 'OpenPorchSF', '2ndFlrSF', 'HalfBath'
]][len(df_train_copy)::]
X_train.info()
X_train.head(5)
X_test.info()
X_test.head(5)
# LinearRegression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
Y_pred_train = lin_reg.predict(X_train)
Y_pred_test = lin_reg.predict(X_test)

# Convert back to value before apply nature logarith
Y_pred_test = np.expm1(Y_pred_test) 

train_loss = np.sqrt( ((Y_train - Y_pred_train)**2).sum() / len(Y_train))
print ("train loss: ", train_loss)
submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": Y_pred_test
    })
submission.to_csv('submission.csv', index=False)
