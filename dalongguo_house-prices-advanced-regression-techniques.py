import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

raw_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#display all columns

pd.set_option('display.max_columns', None)

#display all rows

pd.set_option('display.max_rows', None)

raw_train.head()
raw_train.describe(include='all').T
raw_test.describe(include='O').T.shape
from scipy import stats
sns.distplot(raw_train['SalePrice'],color='blue')
stats.probplot(raw_train['SalePrice'],plot=plt)
#heatmap

fig = plt.figure(figsize=(16,9))

sns.heatmap(raw_train.corr(),cmap='Blues')
#TotalBsmtSF  1stFlrSF GarageYrBlt YearBuilt GrLivArea TotRmsAbvGrd GarageCars GarageArea OverallQual

sns.heatmap(raw_train[['SalePrice','TotalBsmtSF','1stFlrSF' ,'GarageYrBlt' ,'YearBuilt' ,'GrLivArea' ,'TotRmsAbvGrd','GarageCars','GarageArea' ,'OverallQual']].corr(),annot=True)
columns=['GrLivArea','GarageCars','OverallQual','YearBuilt','TotalBsmtSF','SalePrice']

sns.pairplot(raw_train[columns])
#Relationship with numerical variables

#GrLiveArea and Saleprise

data=pd.concat([raw_train['SalePrice'],raw_train['GrLivArea']],axis=1)

data.plot.scatter('GrLivArea','SalePrice',ylim=(0,800000),xlim=(0,6000))
#MSZoning  and Saleprise

data=pd.concat([raw_train['SalePrice'],raw_train['TotalBsmtSF']],axis=1)

data.plot.scatter('TotalBsmtSF','SalePrice')
#Relationship with categorical features

#GarageCars and SalePrice

data=pd.concat([raw_train['SalePrice'],raw_train['GarageCars']],axis=1)

fig=sns.boxplot(x='GarageCars',y='SalePrice',data=data)

fig.axis(ymin=0, ymax=800000)
#OverallQual and SalePrice

data=pd.concat([raw_train['SalePrice'],raw_train['OverallQual']],axis=1)

fig=sns.boxplot(x='OverallQual',y='SalePrice',data=data)

fig.axis(ymin=0, ymax=800000)
#YearBuilt and SalePrice

data=pd.concat([raw_train['SalePrice'],raw_train['YearBuilt']],axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='YearBuilt',y="SalePrice", data=data)

fig.axis(ylim=0,ymax=800000)
n1=raw_train.isnull().sum().sort_values(ascending=False)

n2=raw_test.isnull().sum().sort_values(ascending=False)

null_features=pd.concat([n1[n1!=0],n2[n2!=0]],axis=1, keys=['Train', 'Test'],sort=False)



train_percent=pd.DataFrame(((n1[n1!=0])*100/len(raw_train)).sort_values(ascending=False),columns=['train_percent'])

test_percent=pd.DataFrame(((n2[n2!=0])*100/len(raw_train)).sort_values(ascending=False),columns=['test_percent'])

null_count=pd.concat([null_features,train_percent,test_percent],axis=1,sort=False)



null_count
#Drop missing data largely and Multicollinearity avriables

raw_train.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','TotRmsAbvGrd','1stFlrSF','GarageYrBlt','GarageArea','TotalBsmtSF','2ndFlrSF'],inplace=True)

raw_test.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','TotRmsAbvGrd','1stFlrSF','GarageYrBlt','GarageArea','TotalBsmtSF','2ndFlrSF'],inplace=True)

raw_test.shape
#fill the null whith average  or mode

col=['GarageCars','BsmtUnfSF','KitchenQual','Exterior1st','SaleType','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','Functional','MSZoning','Utilities','BsmtHalfBath']

for c in col:

    raw_test[c].fillna(method='bfill',inplace=True)



raw_train['LotFrontage'].fillna(raw_train['LotFrontage'].mean(),inplace=True)

raw_test['LotFrontage'].fillna(raw_test['LotFrontage'].mean(),inplace=True)



col1=['GarageCond','GarageType','GarageFinish','GarageQual','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtCond','BsmtQual','MasVnrArea','MasVnrType']

for c in col1:

    raw_train[c].fillna(raw_train[c].mode()[0],inplace=True)

    raw_test[c].fillna(raw_test[c].mode()[0],inplace=True)



# remember mode()[0] or it still is nan
train=raw_train

test=raw_test# avoid occur some mistakes ,we can't go back



train.dropna(inplace=True)



print('null:',train.isnull().sum().max())

print('null:',test.isnull().sum().max())# 0 null value
X_train=train.iloc[:,1:-1]

y_train=train.iloc[:,-1]

X_test=test.iloc[:,1:]

X_train.shape,X_test.shape#the foermat must be same,we got it!
#we should find out these categorical features

categorical=X_train.describe(include='O').T

c_name=categorical.index

c_name

from sklearn.preprocessing import LabelEncoder

for c in c_name:

    labelencoder=LabelEncoder()

    X_train[c]=labelencoder.fit_transform(X_train[c])

    X_test[c]=labelencoder.fit_transform(X_test[c])
tr_numeric_feats = X_train.dtypes[X_train.dtypes != "object"].index

te_numeric_feats = X_test.dtypes[X_test.dtypes != "object"].index

from scipy.stats import norm, skew 

tr_skewed_feats = X_train[tr_numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)

te_skewed_feats = X_test[te_numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
tr_skewed_feats=tr_skewed_feats[tr_skewed_feats > 0.75].index

te_skewed_feats=te_skewed_feats[te_skewed_feats > 0.75].index



X_train[tr_skewed_feats]=np.log1p(X_train[tr_skewed_feats])

X_test[te_skewed_feats]=np.log1p(X_test[te_skewed_feats])

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split 
# data standardization except y

X_train = StandardScaler().fit_transform(X_train)

X_test=StandardScaler().fit_transform(X_test)
X_tr,X_te,y_tr,y_te=train_test_split(X_train,y_train,random_state=3)
# method1:xgboox it work wery well!

import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(X_train,y_train)

result=model_xgb.predict(X_test)
#method2:RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

model_randomFore = RandomForestRegressor(n_estimators=400)

model_randomFore.fit(X_train,y_train)

y_pred=model_randomFore.predict(X_tr)
#method3:SVC

from sklearn.svm import SVC

model_svc = SVC(kernel = 'sigmoid', random_state = 3)

model_svc.fit(X_train, y_train)

result=model_svc.predict(X_test)
#method4:DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

model_dct = DecisionTreeClassifier(criterion = 'entropy', random_state = 19)

model_dct.fit(X_train, y_train)

result=model_dct.predict(X_test)
pre=pd.DataFrame(result,columns=['SalePrice'])

submmison=pd.concat([raw_test['Id'],pre],axis=1)

submmison.to_csv('./sub.csv',index=False)