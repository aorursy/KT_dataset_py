#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import files
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
## check for null values
puste=df.isnull().sum().sort_values()
puste.tail(11)
#puste_test=df_test.isnull().sum().sort_values()
#puste_test.tail(33)
sns.heatmap(df.isnull(),yticklabels=False, cbar=False)
#remove high null count column
df.drop(['Alley'], inplace=True, axis =1)
df.info

#remove second high null count column
df.drop(['GarageYrBlt'], inplace=True,axis=1)
df.drop(['FireplaceQu'], inplace=True, axis=1)
#remove columns from df_test
df_test.drop(['FireplaceQu'], inplace=True, axis=1)
df_test.drop(['GarageYrBlt'], inplace=True,axis=1)
df_test.drop(['Alley'], inplace=True, axis =1)
# General data cleaning.
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean()) # filling the null values of LotFrontage with mean values for the column
df_test['MSZoning']=df_test['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])

# filling Pool, fence, misc feature Nan to 0
df['PoolQC']=df['PoolQC'].fillna(0)
df['Fence']=df['Fence'].fillna(0)
df['MiscFeature']=df['MiscFeature'].fillna(0)

# finishing data cleaning
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

## re-check for null values
puste=df.isnull().sum().sort_values()
puste.tail(10)

#no null values
# now dealing with the test data cleaning
df_test['GarageType']=df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].mode()[0])
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mode()[0])
df_test['KitchenQual']=df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['Exterior1st']=df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd']=df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['SaleType']=df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mode()[0])
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mode()[0])
df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mode()[0])
df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mode()[0])
df_test['BsmtHalfBath']=df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].mode()[0])
df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mode()[0])
df_test['BsmtFinType1']=df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])
df_test['BsmtFinType2']=df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_test['BsmtQual']=df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['BsmtCond']=df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['Utilities']=df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['Functional']=df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mode()[0])
df_test['MasVnrType']=df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['GarageCond']=df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test['GarageQual']=df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])
df_test['GarageFinish']=df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['PoolQC']=df_test['PoolQC'].fillna(0)
df_test['Fence']=df_test['Fence'].fillna(0)
df_test['MiscFeature']=df_test['MiscFeature'].fillna(0)


puste_test=df_test.isnull().sum().sort_values()
puste_test.tail(5)


#now we have no null values
#heatmap matrix of correlations
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_test.shape
#end of code - submission part
predictions = pd.DataFrame(y_pred) # transforming into dataframe
submissions_dataframe=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([submissions_dataframe['Id'], predictions], axis=1)
datasets.columns=['Id', 'SalePrice']
datasets.to_csv('sample_submission.csv', index = False)
                    