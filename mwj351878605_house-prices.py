import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
submission_test = pd.read_csv('../input/test.csv')
df = pd.read_csv('../input/train.csv')
df.head()
data = {'df':df,'submission':submission_test}
null_count = pd.DataFrame([])
for key, value in data.items():
    null_count[key] = value.isnull().sum()
null_count.loc[(null_count.df!=0) | (null_count.submission!=0)]
null_count.loc[(null_count.df!=0) | (null_count.submission!=0)].index
drop_col = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'] 
for key, value in data.items():
    value.drop(drop_col,axis=1,inplace=True)
dir_col = {'mean':['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
                  'GarageArea'],
           'mode':['MSZoning','Utilities','Exterior1st','Exterior2nd','Electrical','KitchenQual','Functional',
                  'GarageCars','SaleType'],
           'normal':['GarageYrBlt','LotFrontage','MasVnrArea'],
           'pad':['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                 'GarageType','GarageFinish','GarageQual','GarageCond']}

def completion_col(dir_col,value):
    for name,col in dir_col.items():
        if name=='mode':
            for i in col:
                data = value[i].copy()
                data[data.isnull()] = data.value_counts().idxmax()
                value[i] = pd.DataFrame(data)
        elif name=='mean':
            for i in col:
                data = value[i].copy()
                data[data.isnull()] = round(data.mean(),0)
                value[i] = pd.DataFrame(data)
        elif name=='normal':
            for i in col:
                mean_num = value[i].mean()
                std_num = value[i].std()
                null_count = value[i].isnull().sum()
                null_random_list = np.random.randint(mean_num-std_num,mean_num+std_num,null_count)
                data = value[i].copy()
                data[data.isnull()] = null_random_list
                value[i] = pd.DataFrame(data)
        elif name=='pad':
            for i in col:
                value[i] = value[i].fillna(method='pad')
    return value

df = completion_col(dir_col,df)
submission_test = completion_col(dir_col,submission_test)
sns.distplot(df.SalePrice)
plt.show()
plt.figure(figsize=(10,10))
sns.jointplot(y='SalePrice',x='GrLivArea',data=df)
plt.show()
plt.figure(figsize=(10,10))
sns.jointplot(y='SalePrice',x='TotalBsmtSF',data=df)
plt.show()
plt.figure(figsize=(10,8))
sns.boxplot(y='SalePrice',x='OverallQual',data=df)
plt.show()
plt.figure(figsize=(16,8))
sns.boxplot(y='SalePrice',x='YearBuilt',data=df)
plt.show()
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr,square=True)
plt.show()
a = corr.nlargest(10, 'SalePrice')['SalePrice'].index
b = df[a].corr()
plt.figure(figsize=(15,10))
sns.heatmap(b,fmt='.2f',square=True,annot=True,annot_kws={'size': 15})
plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data=df,vars=cols)
plt.show()
