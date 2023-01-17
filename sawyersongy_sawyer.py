import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_test.info()
df_test.shape
df_train.shape
num=df_train.select_dtypes(exclude='object')
numcorr=num.corr()
plt.subplots(figsize=(20,1))
numcorr.sort_values(by = ['SalePrice'], ascending=False)
sns.heatmap(numcorr.sort_values(by = ['SalePrice'], ascending=False).head(1),cmap='Blues')
numcorr['SalePrice'].sort_values(ascending = False).to_frame()
df_combined = pd.concat((df_test, df_train), sort = False).reset_index(drop = True)
df_combined.drop(['Id'], axis=1, inplace=True)
df_combined.shape
df_combined.isnull().sum().sort_values(ascending = False)
num = df_combined.select_dtypes(exclude = 'object')
num
num.isnull().sum().sort_values(ascending = False)
df_combined['LotFrontage'] = df_combined['LotFrontage'].fillna(df_combined.LotFrontage.median())
df_combined['GarageYrBlt'] = df_combined['GarageYrBlt'].fillna(df_combined.GarageYrBlt.median())
df_combined['MasVnrArea']  = df_combined['MasVnrArea'].fillna(0)  

df_combined.select_dtypes(exclude = 'object').isnull().sum().sort_values(ascending = False)
df_combined[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','GarageArea','GarageCars','BsmtUnfSF','TotalBsmtSF','BsmtFinSF2']] = df_combined[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','GarageArea','GarageCars','BsmtUnfSF','TotalBsmtSF','BsmtFinSF2']].fillna(0)
df_combined.select_dtypes(exclude = 'object').isnull().sum().sort_values(ascending = False)
allna = df_combined.isnull().sum() / len(df_combined)*100
MV = df_combined[allna.index.to_list()]
allna
catmv = MV.select_dtypes(include='object')
catmv.isnull().sum()
catmv.isnull().sum().sort_values()
###for few missing values
df_combined['Electrical']=df_combined['Electrical'].fillna(method='ffill')
df_combined['SaleType']=df_combined['SaleType'].fillna(method='ffill')
df_combined['KitchenQual']=df_combined['KitchenQual'].fillna(method='ffill')
df_combined['Exterior1st']=df_combined['Exterior1st'].fillna(method='ffill')
df_combined['Exterior2nd']=df_combined['Exterior2nd'].fillna(method='ffill')
df_combined['Functional']=df_combined['Functional'].fillna(method='ffill')
df_combined['Utilities']=df_combined['Utilities'].fillna(method='ffill')
df_combined['MSZoning']=df_combined['MSZoning'].fillna(method='ffill')
df_combined.select_dtypes(include='object').isnull().sum().sort_values()
for col in df_combined.columns:
    if df_combined[col].dtype == 'object':
        df_combined[col] =  df_combined[col].fillna('None')        
df_combined.select_dtypes(include='object')
cat = df_combined.select_dtypes(include='object')
cat.head()
for col in cat.columns:
    df_combined[col] = df_combined[col].astype('category')
    df_combined[col] = df_combined[col].cat.codes
df_combined
df_combined.info()

df_combined['GarageFinish'].value_counts()
catmv['GarageFinish'].unique()
## test_dataset
df_test1 = df_combined.iloc[:1459]
df_test1
## training_dataset
df_train1 = df_combined.iloc[1459:]
df_train1
df_preprocessed = pd.concat([df_train1, df_test1], axis=0)
df_preprocessed = df_preprocessed.reset_index(drop=True)
df_preprocessed.index.name = "Id"

file = "/kaggle/working/features_Sawyer.csv"
df_preprocessed.to_csv(file)
df_preprocessed
file = "/kaggle/working/features_Sawyer.csv"
load_data = pd.read_csv(file, index_col = 'Id')
load_data
