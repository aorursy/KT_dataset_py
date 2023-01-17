import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
%%time

df=pd.read_csv("../input/train.csv")
df.head()
df.shape
df.columns
def missingvaluescount(df):
    cols=[]
    for i in df.columns :
        if df[i].isnull().sum() >0 :
            cols.append(i)
    print("No of columns having missing values:",len(cols))
    null_values=df[cols].isnull().sum()#.sort_values(ascending=False)
    percentage=null_values/len(df)*100
    miss_values=pd.concat([null_values,percentage],axis=1,keys=['null_values','percentage'])
    print(miss_values)

missingvaluescount(df)
df[['LotFrontage','SalePrice']].head()
df1=df.select_dtypes(include = ['float64', 'int64'])
for i in df1.columns :
    #print(df1['LotFrontage'].corr(df1[i]))
    if (df1['LotFrontage'].corr(df[i]) > 0.4) & (i!='LotFrontage'):
        print(i,df1['LotFrontage'].corr(df[i]))
df['sqr_1stFlrSF']=df['1stFlrSF']*df['1stFlrSF']
#df['SqrtLotArea']=np.sqrt(df['1stFlrSF'])

df['LotFrontage'].corr(df['sqr_1stFlrSF'])
#df
#df1['LotFrontage'].fillna(df1['1stFlrSF'].mean(),inplace=True)
#df1['LotFrontage'].corr(df1['1stFlrSF'])
temp=df['LotFrontage'].isnull()
#temp
df.LotFrontage[temp]=df.sqr_1stFlrSF[temp]
df.shape
df['LotFrontage'].isnull().sum()
df['Alley'].value_counts()
temp1=df['Alley'].notnull()
df.Alley[temp1].head()
df['Alley'].fillna('None',inplace=True)
df['MasVnrType'].describe()
df['MasVnrType'].value_counts()
df['MasVnrType'].fillna('None',inplace=True)
df['MasVnrType'].isnull().sum()
df['MasVnrArea'].describe()
sns.distplot(df['MasVnrArea'].dropna())
df['MasVnrArea'].fillna(0.0,inplace=True)
df['MasVnrArea'].isnull().sum()
df['BsmtQual'].describe()
df['BsmtQual'].value_counts()
#df.isnull()
bsmnt=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1' ,'BsmtFinType2']
#df[bsmnt].isnull()
df[bsmnt][df['BsmtQual'].isnull()==True]
bsmnttyp2_null=df[df['BsmtFinType2'].isnull()].index.tolist()
len(bsmnttyp2_null)
df['BsmtQual'].iloc[bsmnttyp2_null].isnull().sum()
df[bsmnt].dtypes
df[bsmnt].iloc[bsmnttyp2_null]
df['BsmtQual'].fillna('None',inplace=True)

df['BsmtCond'].fillna('None',inplace=True)

df['BsmtExposure'].fillna('None',inplace=True)

df['BsmtFinType1'].fillna('None',inplace=True)

df['BsmtFinType2'].fillna('None',inplace=True)
df[bsmnt].isnull().sum()
df['Electrical'].isnull().sum()
df['Electrical'].value_counts()
df['Electrical'].fillna('SBrkr',inplace=True)
print(df['Electrical'].value_counts(),"\n",df['Electrical'].isnull().sum())
df['FireplaceQu'].dtype
df['FireplaceQu'].value_counts()
fire=df[df['FireplaceQu'].isnull()].index.tolist()
df_fire=pd.DataFrame(df['Fireplaces'].iloc[fire])
df_fire['Fireplaces'].value_counts()
df['FireplaceQu'].fillna('None',inplace=True)
df['FireplaceQu'].value_counts()
df['FireplaceQu'].isnull().sum()
df['GarageType'].dtype
df['GarageType'].value_counts()
grg=['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
def grg_fun() :
    for i in grg:
        #print("data type: ",df[i].dtype)
        print("No. of Missing values in ",i,"are",df[i].isnull().sum()," and datatype is ",df[i].dtype)
grg_fun()
for i in grg:
    if i!='GarageYrBlt':
        print(df[i].value_counts(),"\n")
grg_index=df[df['GarageType'].isnull()].index.tolist()
len(grg_index)
df[grg].iloc[grg_index].head()
df[['GarageCars','GarageArea']].iloc[grg_index].head()
for i in grg :
    if df[i].dtype=='object' :
        df[i].fillna('None',inplace=True)
    else :
        df[i].fillna(0.0,inplace=True)
grg_fun()
df['PoolQC'].value_counts()
pool=df[df['PoolQC'].isnull()].index.tolist()
df[['PoolQC','PoolArea']].iloc[pool].head()
df['PoolQC'].fillna('None',inplace=True)
df['PoolQC'].value_counts()
df['Fence'].describe()
df['Fence'].isnull().sum()
df['Fence'].fillna('None',inplace=True)
df['MiscFeature'].fillna('None',inplace=True)
df.head()

df.columns
df.drop('sqr_1stFlrSF',axis=1,inplace=True)
df.columns
df.shape
df.isnull().sum()