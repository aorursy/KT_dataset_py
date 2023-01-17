import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
pd.set_option('max_colwidth', 256)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
df.head()
corr = df.corr()
top_feature = corr[abs(corr['SalePrice']>0.5)]
top_feature['SalePrice']
var = 'OverallQual'
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
var = 'YearBuilt'
f, ax = plt.subplots(figsize=(20,20))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("Find most important features relative to target")
corr = df.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice
corr = df.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)
top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(df[col], height=3, kind='reg')
df.isnull().sum().sort_values(ascending= False)[:15]
df['LotFrontage'] =df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mean())
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mode()[0])
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mode()[0])
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mode()[0])
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0])
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].mode()[0])
df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].mean())
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df.drop(['Id'],inplace= True,axis=1)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])
df.isnull().sum().sort_values(ascending= False)[:15]
cols = df.columns
num_cols = df._get_numeric_data().columns
num_cols
col_list = list(set(cols) - set(num_cols))
col_list
y = df.iloc[:, -1].values
y
df1 = df.copy()
df1 = pd.get_dummies(df, columns = ['GarageCond',
 'BsmtExposure',
 'BldgType',
 'GarageQual',
 'KitchenQual',
 'Condition1',
 'Electrical',
 'LotShape',
 'LandSlope',
 'BsmtCond',
 'LotConfig',
 'BsmtFinType2',
 'CentralAir',
 'Neighborhood',
 'HouseStyle',
 'MasVnrType',
 'PavedDrive',
 'SaleType',
 'Condition2',
 'LandContour',
 'ExterQual',
 'HeatingQC',
 'BsmtFinType1',
 'Functional',
 'GarageType',
 'Utilities',
 'GarageFinish',
 'RoofMatl',
 'Exterior1st',
 'BsmtQual',
 'RoofStyle',
 'Heating',
 'Street',
 'ExterCond',
 'Foundation',
 'Exterior2nd',
 'SaleCondition',
 'MSZoning'], drop_first = True)

df1.head()
col = df1.pop("SalePrice")
df1
y
df2 = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df3 = df2.copy()
df2.drop(['PoolQC','Fence','MiscFeature','FireplaceQu','Alley','GarageYrBlt'],axis=1,inplace=True)
df2['LotFrontage'] = df2['LotFrontage'].fillna(df2['LotFrontage'].mean())
df2['GarageType'] = df2['GarageType'].fillna(df2['GarageType'].mode()[0])
df2['GarageFinish'] = df2['GarageFinish'].fillna(df2['GarageFinish'].mode()[0])
df2['MSZoning'] = df2['MSZoning'].fillna(df2['MSZoning'].mode()[0])
df2['BsmtQual'] = df2['BsmtQual'].fillna(df2['BsmtQual'].mode()[0])
df2['BsmtCond'] = df2['BsmtCond'].fillna(df2['BsmtCond'].mode()[0])
df2['BsmtExposure'] = df2['BsmtExposure'].fillna(df2['BsmtExposure'].mode()[0])
df2['KitchenQual'] = df2['KitchenQual'].fillna(df2['KitchenQual'].mode()[0])
df2['Functional'] = df2['Functional'].fillna(df2['Functional'].mode()[0])
df2['GarageQual'] = df2['GarageQual'].fillna(df2['GarageQual'].mode()[0])
df2['GarageCond'] = df2['GarageCond'].fillna(df2['GarageCond'].mode()[0])
df2['GarageCars'] = df2['GarageCars'].fillna(df2['GarageCars'].mode()[0])
df2['GarageArea'] = df2['GarageArea'].fillna(df2['GarageArea'].mean())
df2['SaleType'] = df2['SaleType'].fillna(df2['SaleType'].mode()[0])
df2['BsmtFinType1'] = df2['BsmtFinType1'].fillna(df2['BsmtFinType1'].mode()[0])
df2['BsmtFinSF1'] = df2['BsmtFinSF1'].fillna(df2['BsmtFinSF1'].mode()[0])
df2['BsmtFinSF2'] = df2['BsmtFinSF2'].fillna(df2['BsmtFinSF2'].mode()[0])
df2['BsmtFinType2'] = df2['BsmtFinType2'].fillna(df2['BsmtFinType2'].mode()[0])
df2['Exterior1st'] = df2['Exterior1st'].fillna(df2['Exterior1st'].mode()[0])
df2['Exterior2nd'] = df2['Exterior2nd'].fillna(df2['Exterior2nd'].mode()[0])
df2['BsmtUnfSF'] = df2['BsmtUnfSF'].fillna(df2['BsmtUnfSF'].mode()[0])
df2['TotalBsmtSF'] = df2['TotalBsmtSF'].fillna(df2['TotalBsmtSF'].mode()[0])
df2['BsmtHalfBath'] = df2['BsmtHalfBath'].fillna(df2['BsmtHalfBath'].mode()[0])
df2['BsmtFullBath'] = df2['BsmtFullBath'].fillna(df2['BsmtFullBath'].mode()[0])
df2['Utilities'] = df2['Utilities'].fillna(df2['Utilities'].mode()[0])
df2['MasVnrType'] = df2['MasVnrType'].fillna(df2['MasVnrType'].mode()[0])
df2['MasVnrArea'] = df2['MasVnrArea'].fillna(df2['MasVnrArea'].mode()[0])
ID = df2['Id']
df2.drop(['Id'],axis=1,inplace=True)
df2.head()
cols = df2.columns
num_cols = df2._get_numeric_data().columns
num_cols
col_list1 = list(set(cols) - set(num_cols))
col_list1
df3 = pd.get_dummies(df2, columns = ['Street',
 'Neighborhood',
 'LotShape',
 'RoofMatl',
 'Heating',
 'BsmtExposure',
 'BsmtCond',
 'ExterCond',
 'Electrical',
 'RoofStyle',
 'Condition2',
 'SaleCondition',
 'LotConfig',
 'KitchenQual',
 'GarageFinish',
 'PavedDrive',
 'SaleType',
 'Utilities',
 'GarageQual',
 'Exterior1st',
 'GarageCond',
 'Exterior2nd',
 'Functional',
 'LandSlope',
 'BsmtFinType2',
 'BsmtFinType1',
 'ExterQual',
 'BldgType',
 'BsmtQual',
 'CentralAir',
 'LandContour',
 'Condition1',
 'MSZoning',
 'MasVnrType',
 'GarageType',
 'Foundation',
 'HeatingQC',
 'HouseStyle']
, drop_first = True)
for i in df1.columns:
    if i not in df3.columns:
        df1.drop({i},axis= 1, inplace= True)
        
for i in df3.columns:
    if i not in df1.columns:
        df3.drop({i},axis= 1, inplace= True)
X = df1.iloc[:].values
X_test = df3.iloc[:].values
X.shape
X_test.shape
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X, y)
predrd = regressor.predict(X_test)
predrd