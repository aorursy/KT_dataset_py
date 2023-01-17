import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import IPython
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test = test
data_desc = open('../input/house-prices-advanced-regression-techniques/data_description.txt')
print(data_desc.read())
df_train.head()
df_train.describe().transpose()
df_test.describe().transpose()
# Set target variable
target = df_train['SalePrice']
# Initial split into categorical and numeric variables
categorical_var = df_train.columns[df_train.dtypes.values == object]
numeric_var = df_train.columns[df_train.dtypes.values != object]
categorical_var
numeric_var
# Check for missing values and percentages
(100*df_train.isnull().sum().sort_values(ascending=False)/len(df_train)).head(20)
(100*df_test.isnull().sum().sort_values(ascending=False)/len(df_test)).head(20)
# Drop missing unimportant features
df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','MasVnrType','MasVnrArea'],axis=1,inplace=True)
df_test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','MasVnrType','MasVnrArea'],axis=1,inplace=True)

categorical_var = categorical_var.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MasVnrType'])
numeric_var = numeric_var.drop(['LotFrontage','MasVnrArea'])
# Get highest correlations
df_train.corr()[(df_train.corr()['SalePrice']>0.4)]['SalePrice'].drop('SalePrice').sort_values(ascending=False)
df_train['GarageCars'].corr(df_train['GarageArea'])
# Can drop either one (dropped Garage Area)
df_train.drop('GarageArea',axis=1,inplace=True)
df_test.drop('GarageArea',axis=1,inplace=True)
df_train['GarageCars'].corr(df_train['GarageYrBlt'])
df_train['GrLivArea'].corr(df_train['TotalBsmtSF'])
df_train['GrLivArea'].corr(df_train['1stFlrSF'])
df_train['GrLivArea'].corr(df_train['TotRmsAbvGrd'])
# Drop TotRmsAbvGrd
df_train.drop('TotRmsAbvGrd',axis=1,inplace=True)
df_test.drop('TotRmsAbvGrd',axis=1,inplace=True)
df_train['GrLivArea'].corr(df_train['Fireplaces'])
# Dealing with missing values
# Start from top of list (Garage)
sns.countplot(x='GarageQual',data=df_train)
sns.countplot(x='GarageCond',data=df_train)
sns.countplot(x='GarageFinish',data=df_train)
sns.countplot(x='GarageType',data=df_train)
df_train['YearBuilt'].corr(df_train['GarageYrBlt'])
# Drop GarageYrBlt since high corr with house year built
df_train.drop('GarageYrBlt',axis=1,inplace=True)
df_test.drop('GarageYrBlt',axis=1,inplace=True)
# Drop cond since present in train and test and has similar distribution to GarageQual and probably correlated
df_train.drop('GarageCond',axis=1,inplace=True)
df_test.drop('GarageCond',axis=1,inplace=True)
garagequal_norm = df_train.GarageQual.value_counts(normalize=True)
# print(garagequal_norm)
df_train.loc[df_train['GarageQual'].isnull(),'GarageQual'] = np.random.choice(garagequal_norm.index, size=len(df_train[df_train['GarageQual'].isnull()]),p=garagequal_norm.values)
# df_train['GarageQual'].isnull().sum()
df_test.loc[df_test['GarageQual'].isnull(),'GarageQual'] = np.random.choice(garagequal_norm.index, size=len(df_test[df_test['GarageQual'].isnull()]),p=garagequal_norm.values)
# df_test['GarageQual'].isnull().sum()
garagefinish_norm = df_train.GarageFinish.value_counts(normalize=True)
# print(garagefinish_norm)
df_train.loc[df_train['GarageFinish'].isnull(),'GarageFinish'] = np.random.choice(garagefinish_norm.index, size=len(df_train[df_train['GarageFinish'].isnull()]),p=garagefinish_norm.values)
# df_train['GarageFinish'].isnull().sum()
df_test.loc[df_test['GarageFinish'].isnull(),'GarageFinish'] = np.random.choice(garagefinish_norm.index, size=len(df_test[df_test['GarageFinish'].isnull()]),p=garagefinish_norm.values)
# df_test['GarageFinish'].isnull().sum()
garagetype_norm = df_train.GarageType.value_counts(normalize=True)
# print(garagetype_norm)
df_train.loc[df_train['GarageType'].isnull(),'GarageType'] = np.random.choice(garagetype_norm.index, size=len(df_train[df_train['GarageType'].isnull()]),p=garagetype_norm.values)
# df_train['GarageType'].isnull().sum()
df_test.loc[df_test['GarageType'].isnull(),'GarageType'] = np.random.choice(garagetype_norm.index, size=len(df_test[df_test['GarageType'].isnull()]),p=garagetype_norm.values)
# df_test['GarageType'].isnull().sum()
# Check what other data is missing
(100*df_train.isnull().sum().sort_values(ascending=False)/len(df_train)).head(20)
(100*df_test.isnull().sum().sort_values(ascending=False)/len(df_test)).head(20)
# Update remaining variables
categorical_var = df_train.columns[df_train.dtypes.values == object]
numeric_var = df_train.columns[df_train.dtypes.values != object]
categorical_var
numeric_var
# Dealing with missing values (Basement)
sns.countplot(df_train['BsmtFinType1'])
sns.countplot(df_train['BsmtFinType2'])
sns.countplot(df_train['BsmtExposure'])
sns.countplot(df_train['BsmtQual'])
# Drop BsmtCond in favor of BsmtQual since more evenly distributed
df_train.drop('BsmtCond',axis=1,inplace=True)
df_test.drop('BsmtCond',axis=1,inplace=True)
# Keep BsmtFinType1 and drop BsmtFinType2 since similar description
df_train.drop('BsmtFinType2',axis=1,inplace=True)
df_test.drop('BsmtFinType2',axis=1,inplace=True)
basementqual_norm = df_train.BsmtQual.value_counts(normalize=True)
df_train.loc[df_train['BsmtQual'].isnull(),'BsmtQual'] = np.random.choice(basementqual_norm.index, size=len(df_train[df_train['BsmtQual'].isnull()]),p=basementqual_norm.values)
df_test.loc[df_test['BsmtQual'].isnull(),'BsmtQual'] = np.random.choice(basementqual_norm.index, size=len(df_test[df_test['BsmtQual'].isnull()]),p=basementqual_norm.values)
basementfin1_norm = df_train.BsmtFinType1.value_counts(normalize=True)
df_train.loc[df_train['BsmtFinType1'].isnull(),'BsmtFinType1'] = np.random.choice(basementfin1_norm.index, size=len(df_train[df_train['BsmtFinType1'].isnull()]),p=basementfin1_norm.values)
df_test.loc[df_test['BsmtFinType1'].isnull(),'BsmtFinType1'] = np.random.choice(basementfin1_norm.index, size=len(df_test[df_test['BsmtFinType1'].isnull()]),p=basementfin1_norm.values)
basementexp_norm = df_train.BsmtExposure.value_counts(normalize=True)
df_train.loc[df_train['BsmtExposure'].isnull(),'BsmtExposure'] = np.random.choice(basementexp_norm.index, size=len(df_train[df_train['BsmtExposure'].isnull()]),p=basementqual_norm.values)
df_test.loc[df_test['BsmtExposure'].isnull(),'BsmtExposure'] = np.random.choice(basementexp_norm.index, size=len(df_test[df_test['BsmtExposure'].isnull()]),p=basementqual_norm.values)
# Check what other data is missing
(100*df_train.isnull().sum().sort_values(ascending=False)/len(df_train)).head(20)
(100*df_test.isnull().sum().sort_values(ascending=False)/len(df_test)).head(20)
# Drop NA row since only small percentage
df_train.dropna(inplace=True)
# Check for missing values in training set
df_train.isnull().sum().max()
imp = SimpleImputer(strategy='most_frequent')
# Fill in missing data with most frequent items since only small percentage
df_test[:] = imp.fit_transform(df_test)
# Check missing values in test set
df_test.isnull().sum().max()
# Check to see if any columns in test that are not in train
# SalePrice is in train but not in test but will be dropped when training
df_test.columns.difference(df_train.columns)
# Update remaining variables
categorical_var = df_train.columns[df_train.dtypes.values == object]
numeric_var = df_train.columns[df_train.dtypes.values != object]
categorical_var
# Filtering to keep more important variables
sns.countplot(df_train['Condition1'])
sns.countplot(df_train['Condition2'])
# Drop both conditions since majority of data is just one type
sns.countplot(df_train['RoofStyle'])
sns.countplot(df_train['RoofMatl'])
# Drop Roof features since majority of data is just one type
plt.figure(figsize=(12,6))
sns.countplot(df_train['Exterior2nd'])
# Drop Roof features since majority of data is just one type
# Check if correlated to SalePrice
# Looks like price pretty evenly distributed across
plt.figure(figsize=(12,6))
sns.scatterplot(x='Exterior2nd',y='SalePrice',data=df_train)
sns.countplot(df_train['ExterCond'])
sns.countplot(df_train['ExterQual'])
sns.countplot(df_train['SaleType'])
categorical_var = categorical_var.drop(['Street','LotShape','LandContour','Condition1','Condition2','BldgType','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterCond','Foundation','Heating','CentralAir','Electrical','Functional','PavedDrive','SaleCondition','SaleType'])
categorical_var
# Update numeric var for those with >0.4 corr with price
numeric_var = df_train.corr()[(df_train.corr()['SalePrice']>0.4)]['SalePrice'].drop('SalePrice').sort_values(ascending=False).index
numeric_var
# Check for possible dependent variables
sns.boxplot(x='HouseStyle',y='1stFlrSF',data=df_train)
sns.scatterplot(x='TotalBsmtSF',y='1stFlrSF',data=df_train)
df_train.drop('TotalBsmtSF',axis=1,inplace=True)
df_test.drop('TotalBsmtSF',axis=1,inplace=True)
sns.scatterplot(x='YearBuilt',y='YearRemodAdd',data=df_train)
df_train.drop('YearRemodAdd',axis=1,inplace=True)
df_test.drop('YearRemodAdd',axis=1,inplace=True)
sns.scatterplot(x='GrLivArea',y='1stFlrSF',data=df_train)
df_train.drop('1stFlrSF',axis=1,inplace=True)
df_test.drop('1stFlrSF',axis=1,inplace=True)
sns.scatterplot(x='Fireplaces',y='GrLivArea',data=df_train)
df_train.drop('Fireplaces',axis=1,inplace=True)
df_test.drop('Fireplaces',axis=1,inplace=True)
numeric_var = numeric_var.drop(['TotalBsmtSF','YearRemodAdd','1stFlrSF','Fireplaces'])
numeric_var
# Dealing with skewed data
# train_skew = df_train[numeric_var]
# test_skew = df_test[numeric_var]
# train_skew.skew()>1
# test_skew.skew() > 1
# Check both datasets to make sure both have same skewed data
#train_skew['GrLivArea'].skew()
#sns.distplot(train_skew['GrLivArea'])
#train_skew['GrLivArea'] = stats.boxcox(train_skew['GrLivArea'])[0]
# Check to see if it is less skewed
#print(train_skew['GrLivArea'].skew())
#sns.distplot(train_skew['GrLivArea'])
#test_skew['GrLivArea'].skew()
#sns.distplot(test_skew['GrLivArea'])
#test_skew['GrLivArea'] = stats.boxcox(test_skew['GrLivArea'])[0]
#print(test_skew['GrLivArea'].skew())
#sns.distplot(test_skew['GrLivArea'])
# Put transformed data back into dataset
#df_train[numeric_var] = train_skew
#df_test[numeric_var] = test_skew
# Deal with target variable skewdness
#sns.distplot(df_train['SalePrice'])
#df_train['SalePrice'].skew()
# Drop some data for SalePrice to be less skewed
# Select arbitrary point
100 * (sum((df_train['SalePrice']<300000) == True))/len(df_train['SalePrice'])
df_train['SalePrice'][df_train['SalePrice']<300000].skew()
# Since we still have 92% of data we can remove data with SalePrice > 300,000 and reduce skewedness by factor of ~4
df_train = df_train[df_train['SalePrice']<300000]
df_train.shape
df_test.shape
train_set = pd.concat([pd.get_dummies(df_train[categorical_var],drop_first=True),df_train[numeric_var],df_train['SalePrice']],axis=1)
test_set = pd.concat([pd.get_dummies(df_test[categorical_var],drop_first=True),df_test[numeric_var]],axis=1)
train_set.shape
test_set.shape
# Get missing columns in the training test
missing_cols = set(train_set.columns) - set(test_set.columns)
# Add a missing column in test set with default value equal to 0
for i in missing_cols:
    test_set[i] = 0
# Ensure the order of column in the test set is in the same order than in train set
test_set = test_set[train_set.columns]
train_set.shape
test_set.shape
train_set.describe().transpose()
test_set.describe().transpose()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(train_set.drop('SalePrice',axis=1), train_set['SalePrice'], test_size=0.2, random_state=101)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
sns.scatterplot(y_test,predictions)
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Do prediction on real test data
model.fit(train_set.drop('SalePrice',axis=1),train_set['SalePrice'])
predictions = model.predict(test_set.drop('SalePrice',axis=1))
df_pred = pd.DataFrame(test['Id'],columns=['Id','SalePrice'])
df_pred['SalePrice'] = predictions
df_pred = df_pred.set_index('Id')
(df_pred['SalePrice']<0).sum()
df_pred.to_csv('HPP3_Predictions.csv')