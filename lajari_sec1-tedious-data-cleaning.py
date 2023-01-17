import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('max_columns',100, 'max_rows',100)

sns.set(context='notebook', style='whitegrid', palette='deep')

from sklearn.impute import KNNImputer
from IPython.display import display_html

def disp_side(*args):

    html_str='  '

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

dataset = pd.concat([train,test],axis = 0,ignore_index =True,sort=False)



train.shape,test.shape,dataset.shape
dataset.head()
nullcnt = dataset.isnull().sum().to_frame()

nulldf = nullcnt[nullcnt[0]>0].sort_values(0,ascending=False)

nulldf.drop('SalePrice',axis=0,inplace=True)

print('Number of columns containing null:',nulldf.shape[0])

print('Number of columns containing nulls in 1000s :',(nulldf[0]>1000).sum())

print('Number of columns containing nulls in 100s : ',((1000>nulldf[0])&(nulldf[0] >100)).sum())

print('Number of columns containing nulls in 10s :',((100>nulldf[0]) &(nulldf[0] >10)).sum())

print('Number of columns containing nulls less than 10 :',(nulldf[0]<10).sum())

disp_side(nulldf[:12],nulldf[12:24],nulldf[24:])
bsmcols =  [col for col in dataset.columns if 'Bsmt' in col]

dataset[bsmcols].isnull().sum()
dataset[dataset['TotalBsmtSF'] == 0][bsmcols].head()
# if 'TotalBsmtSF' is 0 or not available then apply following strategy

rows = (dataset['TotalBsmtSF'] == 0) | (dataset['TotalBsmtSF'].isnull())

dataset.loc[rows,['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']] = 'NAv'

dataset.loc[rows,['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']] = 0

dataset.loc[rows,['BsmtUnfSF','TotalBsmtSF']] = 0

dataset[bsmcols].isnull().sum()            
# Remaining nulls are in categorical or discrete columns. Let's replace it with mode

remain = ['BsmtCond','BsmtExposure','BsmtFinType2','BsmtQual']

modes = dataset[remain].mode().values.tolist()[0]

mapdict = dict(zip(remain,modes))



dataset.fillna(mapdict,inplace=True)
dataset[bsmcols].isnull().sum()
garcols =  [col for col in dataset.columns if 'Garage' in col]

dataset[garcols].isnull().sum()
garcat = ['GarageCond','GarageFinish','GarageQual','GarageType']

rows = (dataset['GarageArea'] == 0) & (dataset['GarageCars'] == 0) & (dataset[garcat].isnull().all(axis=1))

dataset.loc[rows,garcat] = 'NAv'

dataset.loc[rows,'GarageYrBlt'] = 0

dataset[garcols].isnull().sum()
dataset[dataset[garcols].isnull().any(axis=1)][garcols]
# calculating mode for categorical columns with GarageType 'Detached' 

dataset.loc[dataset['GarageType'] == 'Detchd',['GarageCond','GarageFinish','GarageQual']].mode()
#  For index 2126 and 2576 we will replace null in following ways



# 'GarageYrBlt' -> 'YearBuilt'

dataset.loc[2126,'GarageYrBlt'] = dataset.loc[2126,'YearBuilt']

dataset.loc[2576,'GarageYrBlt'] = dataset.loc[2576,'YearBuilt']



# categorical and discrete columns -> mode calculated as above

dataset.loc[[2126,2576],['GarageCond','GarageQual']] = 'TA'

dataset.loc[[2126,2576],'GarageFinish'] = 'Unf'

dataset.loc[2576,['GarageCars']] = dataset.loc[dataset['GarageType'] == 'Detchd','GarageCars'].mode().values



# numeric col -> mean

dataset.loc[2576,['GarageArea']] = dataset.loc[dataset['GarageType'] == 'Detchd','GarageArea'].mean()

dataset[garcols].isnull().sum()
null_count = dataset.isnull().sum()

nulldf = null_count[null_count>0]

nulldf.drop('SalePrice',axis = 0)
dataset.fillna({'Alley':'NAv','Fence':'NAv'},inplace = True)

dataset.loc[dataset['MasVnrArea'] == 0,'MasVnrType'] = 'NAv'

dataset.loc[dataset['Fireplaces'] == 0,'FireplaceQu'] = 'NAv'

dataset.loc[dataset['MiscVal'] == 0,'MiscFeature'] = 'NAv'

dataset.loc[dataset['PoolArea']==0,'PoolQC'] = 'NAv'
null_count = dataset.isnull().sum()

nulldf = null_count[null_count>0]

nulldf.drop('SalePrice',axis = 0)
# Lets replace nulls in remainin categorical and discrete columns with mode value. 

# These column's null count are less than 5.

remain_cols = ['Electrical','Exterior1st','Exterior2nd','Functional','KitchenQual','MiscFeature','PoolQC','SaleType','Utilities']

modes = dataset[remain_cols].mode()

mapdict = dict(zip(remain_cols,modes))

dataset.fillna(mapdict,inplace=True)
dataset[dataset[['MasVnrArea','MasVnrType']].isnull().any(axis=1)][['MasVnrArea','MasVnrType']]
rows = dataset[['MasVnrType','MasVnrArea']].isnull().all(axis=1)

dataset.loc[rows,['MasVnrType','MasVnrArea']] = 0

# row 2610 where MasVnr is present

dataset.at[2610,'MasVnrType'] = dataset['MasVnrType'].mode()
pd.crosstab(dataset['MSZoning'],dataset['Neighborhood'])
dataset[dataset['MSZoning'].isnull()][['MSZoning','Neighborhood']]
dataset.fillna({'MSZoning':'RM'},inplace=True)

dataset.at[2904,'MSZoning'] = 'RL'
before = dataset['LotFrontage'].copy()

cormat = dataset.corr()['LotFrontage']

# cormat.drop(['SalePrice','LotFrontage'],axis = 0,inplace=True)

cormat_before = cormat[cormat> 0.3].to_frame()





cormat_before



features = dataset.select_dtypes(np.number).columns.drop('SalePrice')



imputer = KNNImputer(n_neighbors=5,weights = 'uniform')

dataset[features] = imputer.fit_transform(dataset[features])
cormat = dataset.corr()['LotFrontage']

# cormat.drop(['SalePrice','LotFrontage'],axis = 0,inplace=True)

cormat_after = cormat[cormat> 0.3].to_frame()



fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

sns.distplot(before,ax=ax1)

sns.distplot(dataset['LotFrontage'],ax=ax2)



disp_side(cormat_before,cormat_after)
cleaned_train = dataset[:1460].copy()

cleaned_test = dataset[1460:].copy()

cleaned_test.drop('SalePrice',axis=1,inplace=True)
nulls = cleaned_train.isnull().sum()

nulls[nulls>0]
nulls = cleaned_test.isnull().sum()

nulls[nulls>0]
# Storing for future use

cleaned_train.to_csv('ctrain.csv',index = False)

cleaned_test.to_csv('ctest.csv',index=False)