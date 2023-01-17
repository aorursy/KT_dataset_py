import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.head()
df_train.shape
df_test.shape
df_train.describe()
df_test.describe()
df_train.columns , df_test.columns
#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(10,10))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''#before log transformation
sns.distplot(df_train['SalePrice']);
fig_saleprice = plt.figure(figsize=(12,5))
result1 = stats.probplot(df_train['SalePrice'],plot = plt)'''
'''#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])'''
'''#after log transformation
sns.distplot(df_train['SalePrice']);
fig_saleprice2 = plt.figure(figsize=(12,5))
result3 = stats.probplot(df_train['SalePrice'],plot = plt)'''
#below code is used to see which column is more correlated to dependent varibale so first ten columns are more correlated compare to other columns
corr = df_train.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]
fig = plt.subplots()
plt.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

fig1= plt.subplots()
plt.scatter(x = df_train['OverallQual'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)
plt.show()
fig2= plt.subplots()
plt.scatter(x = df_train['GarageCars'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageCars', fontsize=13)
plt.show()
fig3= plt.subplots()
plt.scatter(x = df_train['GarageArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()
fig4= plt.subplots()
plt.scatter(x = df_train['TotalBsmtSF'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
fig5= plt.subplots()
plt.scatter(x = df_train['1stFlrSF'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('1stFlrSF', fontsize=13)
plt.show()
fig6= plt.subplots()
plt.scatter(x = df_train['FullBath'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('FullBath', fontsize=13)
plt.show()
fig7= plt.subplots()
plt.scatter(x = df_train['TotRmsAbvGrd'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotRmsAbvGrd', fontsize=13)
plt.show()
fig8= plt.subplots()
plt.scatter(x = df_train['YearBuilt'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('YearBuilt', fontsize=13)
plt.show()
'''#deleting outliers
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
df = df.drop(df[(df['GarageArea']>1200) & (df['SalePrice']<500000)].index)
df = df.drop(df[(df['TotalBsmtSF']>3000) & (df['SalePrice']<700000)].index)
df = df.drop(df[(df['1stFlrSF']>2700) & (df['1stFlrSF']<700000)].index)'''
#scatterplot
sns.set()
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF']
sns.pairplot(df_train[columns], size = 3)
plt.show();
#feature engineering
df_train['TotalSF'] = df_train['TotalBsmtSF']+df_train['1stFlrSF']+df_train['2ndFlrSF']
df_train=df_train.drop(columns={'1stFlrSF', '2ndFlrSF','TotalBsmtSF'})
df_train['wholeExterior'] = df_train['Exterior1st']+df_train['Exterior2nd']
df_train=df_train.drop(columns={'Exterior1st','Exterior2nd'})
df_train['Bsmt'] = df_train['BsmtFinSF1']+ df_train['BsmtFinSF2']
df_train = df_train.drop(columns={'BsmtFinSF1','BsmtFinSF2'})
df_train['TotalBathroom'] = df_train['FullBath'] + df_train['HalfBath']
df_train = df_train.drop(columns={'FullBath','HalfBath'})


df_test['TotalSF'] = df_test['TotalBsmtSF']+df_test['1stFlrSF']+df_test['2ndFlrSF']
df_test=df_test.drop(columns={'1stFlrSF', '2ndFlrSF','TotalBsmtSF'})
df_test['wholeExterior'] = df_test['Exterior1st']+df_test['Exterior2nd']
df_test=df_test.drop(columns={'Exterior1st','Exterior2nd'})
df_test['Bsmt'] = df_test['BsmtFinSF1']+ df_test['BsmtFinSF2']
df_test = df_test.drop(columns={'BsmtFinSF1','BsmtFinSF2'})
df_test['TotalBathroom'] = df_test['FullBath'] + df_test['HalfBath']
df_test = df_test.drop(columns={'FullBath','HalfBath'})
frames = [df_train,df_test]
df = pd.concat(frames,keys=['train','test'])
df
df_missing=df.isnull().sum().sort_values(ascending=False)
df_missing
cat_col = df.select_dtypes(include=['object'])
cat_col.isnull().sum()
cat_col.columns
num_col = df.select_dtypes(include=['int64', 'float64'])
num_col.isnull().sum()
num_col.columns
# handling missing values of numerical columns
df['LotFrontage'] = df['LotFrontage'].fillna(value=0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value=0)
df['MasVnrArea'] = df['MasVnrArea'].fillna(value=0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(value=0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(value=0)
df['GarageArea'] = df['GarageArea'].fillna(value=0)
df['GarageCars'] = df['GarageCars'].fillna(value=0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(value=0)
df['Bsmt'] = df['Bsmt'].fillna(value=0)
df['TotalSF'] = df['TotalSF'].fillna(value=0)
# handling missing values of categorical columns
df['MSZoning'] = df['MSZoning'].fillna(value='None')
df['GarageQual'] = df['GarageQual'].fillna(value='None')
df['GarageCond'] = df['GarageCond'].fillna(value='None')
df['GarageFinish'] = df['GarageFinish'].fillna(value='None')
df['GarageType'] = df['GarageType'].fillna(value='None')
df['BsmtExposure'] = df['BsmtExposure'].fillna(value='None')
df['BsmtCond'] = df['BsmtCond'].fillna(value='None')
df['BsmtQual'] = df['BsmtQual'].fillna(value='None')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(value='None')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(value='None')
df['MasVnrType'] = df['MasVnrType'].fillna(value='None')
df['Utilities'] = df['Utilities'].fillna(value='None')
df['Functional'] = df['Functional'].fillna(value='None')
df['Electrical'] = df['Electrical'].fillna(value='None')
df['KitchenQual'] = df['KitchenQual'].fillna(value='None')
df['SaleType'] = df['SaleType'].fillna(value='None')
df['wholeExterior'] = df['wholeExterior'].fillna(value='None')
df = df.drop(columns={'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'})
df.columns
df_main = pd.get_dummies(df)
df_main
df_main.shape
#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = df_main.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice correlation matrix
k = 40 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_main[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(10,10))
hm = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
EID = df_main.loc['test']
df_test = df_main.loc['test']
df_train = df_main.loc['train']
EID = EID.Id
df_test.drop(['SalePrice','Id'], axis =1, inplace=True)
X_train = df_train.drop(['SalePrice','Id'], axis = 1)
y_train = df_train['SalePrice']
import xgboost
xgboost = xgboost.XGBRegressor(learning_rate=0.05,
                      colsample_bytree = 0.5,
                      subsample = 0.8,
                      n_estimators=1000,
                      max_depth=5,
                      gamma=5)

xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(df_test)
y_pred 
#making main csv file
main_submission = pd.DataFrame({'Id': EID, 'SalePrice': y_pred})

main_submission.to_csv("submission.csv", index=False)
main_submission.head()