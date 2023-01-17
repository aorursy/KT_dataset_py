# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.head()
train_df.info()
train_df = train_df.drop('Id',axis=1)
test_df = test_df.drop('Id',axis =1)
train_df.describe().T
cat_cols = train_df.select_dtypes("object").columns
int_cols = train_df.select_dtypes("int64").columns
float_cols = train_df.select_dtypes("float64").columns
num_cols = int_cols.append(float_cols)
print(num_cols)
print("No of numerical columns : ",len(num_cols))

print(cat_cols)
print("No of categorical columns : ",len(cat_cols))

missing_train = pd.DataFrame(train_df.isnull().sum(),columns=['Count'])
missing_test = pd.DataFrame(test_df.isnull().sum(),columns=['Count'])
missing_train = missing_train[missing_train['Count']>0]
missing_train['Percentage'] = (missing_train['Count']/len(train_df)) * 100
missing_train = missing_train.sort_values(by='Percentage',ascending =False)
missing_train
missing_test = missing_test[missing_test['Count']>0]
missing_test['Percentage'] = (missing_test['Count']/len(train_df)) * 100
missing_test = missing_test.sort_values(by='Percentage',ascending =False)
missing_test
set(missing_train.index).difference(set(missing_test.index))
set(missing_test.index).difference(set(missing_train.index))
test_df['SalePrice'] = np.nan
df = pd.concat([train_df,test_df],axis=0,ignore_index=True)
df.head()
df.shape
missing = pd.DataFrame(df.isnull().sum(),columns=['Count'])
missing = missing[missing['Count']>0]
missing['Percentage'] = (missing['Count']/len(df)) * 100
missing = missing.sort_values(by='Percentage',ascending =False)
missing
missing.drop('SalePrice',axis=0,inplace = True)
missing_cat=set(cat_cols).intersection(set(missing.index))
missing_cat
missing_num=set(num_cols).intersection(set(missing.index))
missing_num
len(missing_num)
missing_cat
df['Alley'].value_counts(dropna=False,normalize=True)
df['Alley'].fillna('No_Alley_Access',inplace=True)
df['BsmtCond'].value_counts(dropna=False,normalize=True)
df[((df['TotalBsmtSF']!=0) & (df['TotalBsmtSF'].notna())) & ((df['BsmtCond'].isna()) |(df['BsmtCond'].isna())
                                         |(df['BsmtExposure'].isna()) | (df['BsmtFinType1'].isna())
                                         |(df['BsmtFinType2'].isna())  |(df['BsmtQual'].isna()))][[ 'BsmtCond',
                                         'BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','TotalBsmtSF',
                                          'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath' ]]
base_cols = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']
for col in base_cols : 
    val = df[col].mode()
    #print(val[0])
    #print(type(val))
    df.loc[(df['TotalBsmtSF']!=0) & (df['TotalBsmtSF'].notna()) & (df[col].isna()),col] = val[0]
df[((df['TotalBsmtSF']!=0) & (df['TotalBsmtSF'].notna())) & ((df['BsmtCond'].isna()) |(df['BsmtCond'].isna())
                                         |(df['BsmtExposure'].isna()) | (df['BsmtFinType1'].isna())
                                         |(df['BsmtFinType2'].isna())  |(df['BsmtQual'].isna()))][[ 'BsmtCond',
                                         'BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','TotalBsmtSF',
                                          'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath' ]]
df.fillna({'BsmtCond':'No_Basement','BsmtExposure':'No_Basement','BsmtFinType1':'No_Basement','BsmtFinType2':'No_Basement','BsmtQual':'No_Basement'},inplace=True)
df[['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']].isnull().sum()
df.loc[((df['Fireplaces'] != 0) & (df['FireplaceQu'].isna())),'FireplaceQu']
df.fillna({'Fence':'No_Fence','FireplaceQu':'No_Fireplace'},inplace=True)
df[((df['GarageArea'] > 0) | (df['GarageCars'] > 0)) & (df['GarageCond'].isna() | df['GarageFinish'].isna()
                                                           | df['GarageQual'].isna() | df['GarageType'].isna() |
                                                       df['GarageYrBlt'].isna())][['GarageArea','GarageCars','GarageCond',
                                                        'GarageFinish','GarageQual','GarageType','GarageYrBlt']]
gar_cols = ['GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
for col in gar_cols : 
    val = df[col].mode()
    df.loc[(((df['GarageArea'] > 0) | (df['GarageCars'] > 0)) & (df[col].isna())),col] = val[0]
df[((df['GarageArea'] > 0) | (df['GarageCars'] > 0)) & (df['GarageCond'].isna() | df['GarageFinish'].isna()
                                                           | df['GarageQual'].isna() | df['GarageType'].isna() |
                                                       df['GarageYrBlt'].isna())][['GarageArea','GarageCars','GarageCond',
                                                        'GarageFinish','GarageQual','GarageType','GarageYrBlt']]
df.fillna({'GarageCond': 'No_Garage','GarageFinish' : 'No_Garage'
                 ,'GarageQual': 'No_Garage','GarageType' : 'No_Garage'},inplace=True)
df[((df['GarageArea'] > 0) | (df['GarageCars'] > 0)) & (df['GarageCond'].isna() | df['GarageFinish'].isna()
                                                           | df['GarageQual'].isna() | df['GarageType'].isna() |
                                                       df['GarageYrBlt'].isna())][['GarageArea','GarageCars','GarageCond',
                                                        'GarageFinish','GarageQual','GarageType','GarageYrBlt']]
df[(df['GarageYrBlt'].isna()) & (df['GarageArea'] > 0)][['GarageYrBlt','GarageArea']]
df[df['GarageYrBlt'].isna()][['GarageYrBlt','GarageArea']]
df['GarageYrBlt'].fillna(-1,inplace=True)
df[(df['PoolQC'].isna()) & (df['PoolArea'] > 0)][['PoolQC','PoolArea']]
val = df['PoolQC'].mode()
df.loc[((df['PoolArea'] > 0) & (df['PoolQC'].isna())),'PoolQC'] = val[0]
df[(df['PoolQC'].isna()) & (df['PoolArea'] > 0)][['PoolQC','PoolArea']]
df[df['PoolQC'].isna()][['PoolQC','PoolArea']]
df['MiscFeature'].fillna('None',inplace = True)
train_df.fillna({'MiscFeature':'None','PoolQC':'No_Pool'},inplace=True)
train_df['Electrical'].value_counts(dropna=False,normalize=True)
train_df[train_df['Electrical'].isna()][['Electrical','Utilities','CentralAir']]
e_df = train_df[(train_df['Utilities']=='AllPub') & (train_df['CentralAir']=='Y')][['Electrical','Utilities','CentralAir']]
e_df['Electrical'].value_counts(normalize=True)
train_df['Electrical'].fillna('SBrkr',inplace=True)
train_df[train_df['MasVnrType'].isna()][['MasVnrType','MasVnrArea']]
train_df[(train_df['MasVnrType'].notna()) & (train_df['MasVnrType'].isna())][['MasVnrType','MasVnrArea']]
train_df[(train_df['MasVnrType'].isna()) & (train_df['MasVnrType'].notna())][['MasVnrType','MasVnrArea']]
train_df.fillna({'MasVnrType':'None','MasVnrArea':0},inplace=True)
#lot_df = train_df[(train_df['LotFrontage'].isna())][['LotFrontage','Street']]
lot_df = train_df[train_df['LotFrontage'].isna()]
lot_df['MSZoning'].value_counts(normalize=True)
for i in cat_cols:
    print(lot_df[i].value_counts(normalize=True))
    print('\n')

lot_corr=pd.DataFrame(train_df.corr())
lot_corr = lot_corr[lot_corr['LotFrontage']>0.25]
lot_corr['LotFrontage']
train_df['LotFrontage'].median()
train_df['LotFrontage'].fillna(train_df['LotFrontage'].median(),inplace=True)
null_val = train_df.isnull().sum()>0
null_val[null_val == True]




