import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
pwd
%%time
df=pd.read_csv("../input/train.csv")
df[100:110]
df.shape
df.columns
df.describe()
df.SalePrice.describe()
df.dtypes
for i in df.columns :
    if df[i].dtype=='object' :
        print("\n")        
        print(i)
        print(df[i].value_counts())
        print("\n")
#missing_values=df.columns[df.isnull().any()]
#df[missing_values].isnull().sum()
null_values=df.isnull().sum().sort_values(ascending=False)
percentage=null_values/len(df)*100
miss_values=pd.concat([null_values,percentage],axis=1,keys=['null_values','percentage'])
miss_values
plt.figure(figsize=(15,8))
miss_values['percentage'].head(20).plot.bar(color='blue')
#for i in miss_values['percentage'].head(20) :
    #plt.text(i/5,2,'%d' % i, ha='center', va='bottom')
df_eda=df[[col for col in df if df[col].isnull().sum() / len(df) >= 0.2]]
df_eda.columns
df.drop(columns=df_eda.columns, inplace=True)
plt.figure(figsize=(10,8))
sns.distplot(df['SalePrice'],hist=False,kde=True)
df.select_dtypes(include = ['float64', 'int64']).hist(figsize=(16, 20), xlabelsize=8, ylabelsize=8)
df[['1stFlrSF','TotalBsmtSF','LotFrontage','SalePrice']].hist()

df.drop(['Id'],inplace=True,axis=1)
#%%time
#df.select_dtypes(include = ['float64', 'int64']).plot.bar()
#plt.matshow(df.corr())
df.corr()['SalePrice'][:-1].sort_values(ascending=False)
df2=pd.DataFrame(df.select_dtypes(include = ['float64', 'int64']))
df2.iloc[:,:20]
type(df2)
df2.columns
#sns.pairplot(df2,y_vars=['1stFlrSF'],x_vars=['SalePrice','TotalBsmtSF','LotFrontage','3SsnPorch','YrSold'])
%%time
sns.pairplot(df2,y_vars=['SalePrice'],x_vars=['1stFlrSF','TotalBsmtSF','LotFrontage','3SsnPorch','YrSold'])
imp=['1stFlrSF','TotalBsmtSF','LotFrontage']
for i in imp :
    print("Null values in ",i," are: ",df[i].isnull().sum())
#    df[i].isnull().sum()
#df[['LotFrontage','SalePrice']].head()
df.MSZoning.value_counts()
df.MSZoning.isnull().sum()
df_MSZoning=pd.DataFrame()
%%time
df_MSZoning['MSZoning']=df.MSZoning
df_MSZoning['SalePrice']=df.SalePrice
#del df
df_MSZoning.head()
df_MSZoning['MSZoning'].value_counts().plot(kind='bar')
df_MSZoning.groupby(['MSZoning']).mean().plot(kind='bar')
df_MSZoning.groupby(['MSZoning']).describe()
#Saleprice variation for each zone in MSZoning

#df_MSZoning_rl=df_MSZoning[df_MSZoning.MSZoning=='RL']
#df_MSZoning_rm=df_MSZoning[df_MSZoning.MSZoning=='RM']
#df_MSZoning_rh=df_MSZoning[df_MSZoning.MSZoning=='RH']
#df_MSZoning_fv=df_MSZoning[df_MSZoning.MSZoning=='FV']
#df_MSZoning_c=df_MSZoning[df_MSZoning.MSZoning=='C (all)']

#df_MSZoning_rh.columns


#df_MSZoning_rl.sort_values(by=['SalePrice']).reset_index(drop=True).plot(x='MSZoning',y='SalePrice')


#df_MSZoning_rm.sort_values(by=['SalePrice']).reset_index(drop=True).plot(x='MSZoning',y='SalePrice')


#df_MSZoning_rh.sort_values(by=['SalePrice']).reset_index(drop=True).plot(x='MSZoning',y='SalePrice')


#df_MSZoning_fv.sort_values(by=['SalePrice']).reset_index(drop=True).plot(x='MSZoning',y='SalePrice')


#df_MSZoning_c.sort_values(by=['SalePrice']).reset_index(drop=True).plot(x='MSZoning',y='SalePrice')

corr_cols=df.corr()['SalePrice'][:-1].sort_values(ascending=False).head(10).index.tolist()
corr_cols
def nullcount(df,imp) :
    for i in imp :
        print("Null values in ",i," are: ",df[i].isnull().sum())
nullcount(df,corr_cols)
df_train=df[corr_cols]
df_train.head()
df_test=pd.read_csv("../input/test.csv",usecols=corr_cols)
df_test.isnull().sum()
df_test.describe()
#df_test['TotalBsmtSF'].value_counts()
#df_test['GarageCars'].value_counts()
for i in ['TotalBsmtSF','GarageCars','GarageArea'] :
    print(df_test[i].value_counts())
df_test.fillna(0,inplace=True)
df_test.isnull().sum()
df_train.isnull().sum()
df_test=df_test.reindex(sorted(df_test.columns), axis=1)
df_train=df_train.reindex(sorted(df_train.columns), axis=1)
print(df_test.columns,df_test.shape)
print(df_train.columns,df_train.shape)
target=df.SalePrice
target.shape
from sklearn import linear_model 
model=linear_model.LogisticRegression()
model
%%time
model.fit(df_train,target)
model.predict(df_train)
model.score(df_train,target)
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr
%%time
rfr.fit(df_train,target)
rfr.score(df_train,target)
id_col=pd.DataFrame(pd.read_csv("../input/test.csv",usecols=['Id']))
type(id_col)
id_col.shape
id_col.head()

pred=pd.DataFrame(rfr.predict(df_test),columns=['SalePrice'])
pred.head()
pred.shape
submission=id_col.join(pred)
#%%time
#submission.to_csv("../input/RandomForest_10_corr_cols.csv",index=False)
#df.shape

#x=df
#x.drop(['SalePrice'],axis=1,inplace=True)

#x=pd.DataFrame(x.select_dtypes(include = ['float64', 'int64']))

#%%time
#model.fit(x,target)