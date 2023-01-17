import numpy as np

import pandas as pd

import os 
for dirname,_,filenames in os.walk('/content'):

  for filename in filenames:

    print(os.path.join(dirname,filename))

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

train = pd.read_csv('/content/train.csv')

train.head()
test = pd.read_csv('/content/test.csv')

test.head()
sub = pd.read_csv('/content/sample_submission.csv')

sub.head()
train.shape
test.shape
train.info()
df = pd.concat([train,test])

df.shape
df_num = df.select_dtypes(include =(['float64','int64']))

df_num.head()
df_cat =df.select_dtypes(include = ['object'])

df_cat.head()
df_num.describe()
df_cat.describe()
corr =df_num.corr()
plt.figure(figsize=(25,15))

sns.heatmap(corr,cmap='YlGnBu',annot =True)

plt.figure(figsize=(25,15))

df.groupby(['YrSold','MoSold'])['MoSold'].count().plot(kind ='bar')

plt.tight_layout()

plt.show()
sns.countplot('YrSold',data = df)

plt.show()
plt.figure(figsize = (25,15))

sns.countplot('YearBuilt',data = df)
cols = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']

for i in cols :

  plt.figure(figsize=(5,2))

  plt.hist(df[i])

  plt.xlabel('')

  plt.title(i,color= 'g')

  plt.tight_layout()

  plt.show()
## analysing categorical basement columns



cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']



for i in cols:

    plt.figure(figsize=(5,2))

    df.groupby([i])[i].count().plot(kind='bar')

    plt.xlabel('')

    plt.title(i,color='g')

    plt.tight_layout()

    plt.show()
cols= ['GarageYrBlt','GarageCars','GarageArea']

for i in cols:

    plt.figure(figsize=(5,2))

    plt.hist(df[i])

    plt.xlabel('')

    plt.title(i,color='g')

    plt.tight_layout()

    plt.show()
#GrLivArea''SalePrice

data = pd.concat([df['GrLivArea'],df['SalePrice']],axis =1)

data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))


data = pd.concat([df['TotalBsmtSF'],df['SalePrice']],axis =1)

data.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim=(0,800000))
cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

for i in cols:

  plt.figure(figsize=(5,2))

  df.groupby([i])[i].count().plot(kind='bar')

  plt.xlabel(i)

cols= ['GarageYrBlt','GarageCars','GarageArea']

for i in cols:

    plt.figure(figsize=(5,2))

    plt.hist(df[i])

    plt.xlabel('')

    plt.title(i,color='g')

    plt.tight_layout()

    plt.show()
var='GarageArea'

data=pd.concat([df['SalePrice'],df[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))

plt.show()
#OverallQual,SalePrice

data = pd.concat([df['OverallQual'],df['SalePrice']],axis =1)

sns.boxplot(x='OverallQual',y='SalePrice',data = data )
var='YearBuilt'

plt.figure(figsize=(20,5))

data=pd.concat([df['SalePrice'],df[var]],axis=1)

sns.boxplot(x=var,y='SalePrice',data=data)

plt.show()
round((df.isnull().sum()/len(df.index))*100,2).sort_values(ascending=False)
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].median())

list1 = ['BsmtQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','FireplaceQu','MasVnrType', 'MasVnrArea']
for i in list1:

    df[i]=df[i].fillna(df[i].mode()[0])
columns_cat = ['Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 

           'Exterior1st', 'KitchenQual','MSZoning','Electrical']

columns_num = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea']
for i in columns_cat:

    df[i]=df[i].fillna(df[i].mode()[0])

    

for i in columns_num:

    df[i]=df[i].fillna(df[i].median())
round((df.isnull().sum()/len(df.index)*100),2).sort_values(ascending=False)

sns.distplot(df['LotArea'])

plt.show()
## applying log transformation to correct positive skewness in the data



df['LotArea']=np.log(df['LotArea'])

sns.distplot(df['LotArea'])

plt.show()
sns.distplot(train['SalePrice'])

plt.show()



print("Skewness: %f" % df['SalePrice'].skew())

print("Kurtosis: %f" % df['SalePrice'].kurt())



#applying log transformation

train['SalePrice'] = np.log(train['SalePrice'])



sns.distplot(train['SalePrice'])

plt.show()
df['GrLivArea'] = np.log(df['GrLivArea'])



sns.distplot(df['GrLivArea'])

plt.show()
# Feature Engineering



# Create some new features



df['house_age']=df['YrSold']-df['YearBuilt']

df['house_age'].describe()
df.loc[df['house_age']<0]

df.loc[df['house_age']<0,'YrSold']=2009

df['house_age']=df['YrSold']-df['YearBuilt']

df['house_age'].describe()
# Total surface area = 1stflr area + 2ndflr area + bsmt area 

df['TotalSA']=df['1stFlrSF']+df['2ndFlrSF']+df['TotalBsmtSF']



# Total full bathrooms 

df['TotalBath']=df['FullBath']+0.5*df['HalfBath']



# Total full bsmt bathrooms 

df['TotalBsmtBath']=df['BsmtFullBath']+0.5*df['BsmtHalfBath']
df_cat=df.select_dtypes(include=['object'])

df_cat.head()
for i in df_cat.columns:

    print(df_cat[i].value_counts())
df_cat.drop(['Street','Utilities'],axis=1,inplace=True)

df.drop(['Street','Utilities'],axis=1,inplace=True)
## Creating dict to map values later



dict={'Y' : 1, 'N' : 0, 'Ex': 1, 'Gd' : 2, 'TA' :3, 'Fa' : 4, 'Po' : 5,  

     'GLQ' : 1, 'ALQ' : 2, 'BLQ' : 3, 'Rec' : 4, 'LwQ' : 5, 'Unf' : 6, 'NA' :7,

     'Gd' : 1 , 'Av' :2, 'Mn' : 3, 'No' :4, 'Gtl' : 1, 'Mod' : 2, 'Sev' :3,

      'Reg' : 1, 'IR1' :2, 'IR2' :3, 'IR3' :4}



## selecting columns for mapping dict



cols=['KitchenQual','LotShape','LandSlope','HeatingQC','FireplaceQu','ExterQual','ExterCond','BsmtQual',

     'BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','CentralAir']
for i in cols:

    df_cat[i]=df_cat[i].map(dict)
df_cat.head()
df_cat.head()


df_dummies=pd.get_dummies(df_cat,drop_first=True)

df_dummies.head()
for cols in df_cat.columns:

    df.drop([cols],axis=1,inplace=True)
df_final=pd.concat([df,df_dummies],axis=1)



df_final=df_final.loc[:,~df_final.columns.duplicated()]

df_final.shape

df_final.head()

df_test=df_final.iloc[1460:,:]
df_test.drop(['SalePrice'],inplace=True,axis=1)
df_train=df_final.iloc[:1460,:]
X_train=df_train.drop(['SalePrice'],axis=1)

y_train=df_train['SalePrice']
X_train.head()
from sklearn.ensemble import RandomForestRegressor 

rf=RandomForestRegressor()

y_train.head()
from xgboost import XGBRegressor

xgb=XGBRegressor()

xgb.fit(X_train,y_train)
y_pred=xgb.predict(df_test)
sub=pd.concat([test['Id'],pd.DataFrame(y_pred)],axis=1)

sub.columns=['Id','SalePrice']

sub.head()