# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/train.csv')

df_train.head()
df_train.shape
sns.distplot(df_train['SalePrice'])
print('skewness %f' % df_train['SalePrice'].skew())

print('kurtosis %f' % df_train['SalePrice'].kurt())
var='GrLivArea'

data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
var='OverallQual'

data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)

f,ax=plt.subplots(figsize=(8,6))

fig=sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ylim=0,ymax=800000)
var='YearBuilt'

data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)

f,ax=plt.subplots(figsize=(18,6))

fig=sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ylim=0,ymax=800000)

plt.xticks(rotation=90)
k=10 # number of variables for heatmap

corrmat=df_train.corr()

cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm=np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols,xticklabels=cols)

plt.show()
cols.values
sns.set()

sns.pairplot(df_train[cols],size=2.5)

plt.show()
#missing values

total=df_train.isnull().sum().sort_values(ascending=False)

percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total,percent],axis=1,keys=['total','percent'])

missing_data.head()
train=pd.read_csv('/kaggle/input/train.csv')

test=pd.read_csv('/kaggle/input/test.csv')
print(train.shape)

print(test.shape)
train_id=train['Id']

test_id=test['Id']
train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)
train=train.drop(train[(train['GrLivArea']>4000)&(train['GrLivArea']<300000)].index)

fig,ax=plt.subplots()

ax.scatter(train['GrLivArea'],train['SalePrice'])

plt.ylabel('SalePrice',fontsize=13)

plt.xlabel('GrLivArea',fontsize=13)

plt.show()
sns.distplot(train['SalePrice'],fit=norm)

(mu,sigma)=norm.fit(train['SalePrice'])

print('\n mu={:.2f} and sigma={:.2f}\n'.format(mu,sigma))



plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu,sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('PriceSales Distribution')



fig=plt.figure()

res=stats.probplot(train['SalePrice'],plot=plt)

plt.show()
#normalize

train['SalePrice']=np.log(train['SalePrice'])

sns.distplot(train['SalePrice'],fit=norm)

(mu,sigma)=norm.fit(train['SalePrice'])

print('\n mu={:.2f} and sigma={:.2f}\n'.format(mu,sigma))



plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu,sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('PriceSales Distribution')



fig=plt.figure()

res=stats.probplot(train['SalePrice'],plot=plt)

plt.show()
ntrain=train.shape[0]

ntest=test.shape[0]

y_train=train.SalePrice.values

all_data=pd.concat((train,test)).reset_index(drop=True)

all_data.drop(['SalePrice'],axis=1,inplace=True)

all_data.shape

all_data_na=(all_data.isnull().sum()/len(all_data))*100

all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]

missing_data=pd.DataFrame({'missing_data':all_data_na})

missing_data.head()
f,ax=plt.subplots(figsize=(15,12))

plt.xticks(rotation=90)

sns.barplot(x=all_data_na.index,y=all_data_na)

plt.xlabel('Features',fontsize=15)

plt.ylabel('Percent of missing values',fontsize=15)

plt.title('Percent of missing values by features',fontsize=15)
all_data['PoolQC']=all_data['PoolQC'].fillna('None')

all_data['MiscFeature']=all_data['MiscFeature'].fillna('None')

all_data['Alley']=all_data['Alley'].fillna('None')

all_data['Fence']=all_data['Fence'].fillna('None')

all_data['FireplaceQu']=all_data['FireplaceQu'].fillna('None')
all_data['LotFrontage']=all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):

    all_data[col]=all_data[col].fillna('None')

    

for col in ('GarageYrBlt','GarageArea','GarageCars'):

    all_data[col]=all_data[col].fillna(0)

    

for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):

    all_data[col]=all_data[col].fillna(0)   

    

for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):

    all_data[col]=all_data[col].fillna('None')        
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Functional']=all_data['Functional'].fillna('Typ')

all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual']=all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st']=all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd']=all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType']=all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass']=all_data['MSSubClass'].fillna('None')
all_data=all_data.drop(['Utilities'],axis=1)

all_data_na=(all_data.isnull().sum()/len(all_data))*100

all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]

missing_data=pd.DataFrame({'missing_data':all_data_na})

missing_data.head()
all_data['MSSubClass']=all_data['MSSubClass'].apply(str)

all_data['OverallCond']=all_data['OverallCond'].apply(str)

all_data['YrSold']=all_data['YrSold'].apply(str)

all_data['MoSold']=all_data['MoSold'].apply(str)
from sklearn.preprocessing import LabelEncoder

cols=['FireplaceQu','BsmtQual','BsmtCond','GarageQual','GarageCond','ExterQual','ExterCond','HeatingQC','PoolQC','KitchenQual',

     'BsmtFinType1','BsmtFinType2','Functional','Fence','BsmtExposure','GarageFinish','LandSlope','LotShape','PavedDrive',

     'Street','Alley','CentralAir','MSSubClass','OverallCond','YrSold','MoSold']

for c in cols:

    lbl=LabelEncoder()

    lbl.fit(list(all_data[c].values))

    all_data[c]=lbl.transform(list(all_data[c].values))

print (all_data.shape)    
all_data['TotalSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
from scipy.stats import norm,skew



numeric_feats=all_data.dtypes[all_data.dtypes !='object'].index



skewed_feats=all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness=pd.DataFrame({'Skew':skewed_feats})

skewness.head()
skewness=skewness[abs(skewness)>0.5]

print('There are {} skewed numerical features to Box-Cox'.format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features=skewness.index

lam=0.15 # empirical value

for feat in skewed_features:

    all_data[feat]=boxcox1p(all_data[feat],lam)
all_data=pd.get_dummies(all_data)

print(all_data.shape)
train=all_data[:ntrain]

test=all_data[ntrain:]
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin,clone

from sklearn.model_selection import KFold,cross_val_score,train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb
n_folds=5

def rmsle_cv(model):

    kf=KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse=np.sqrt(-cross_val_score(model,train.values,y_train,scoring='neg_mean_squared_error',cv=kf))

    return (rmse)
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))
ENet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,random_state=3,l1_ratio=.9))
KRR=KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5)
GBoost=GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=4,max_features='sqrt',

                                min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=5)
model_xgb=xgb.XGBRegressor(colsample_bytree=0.4603,gamma=0.0468,learning_rate=0.05,max_depth=3,min_child_weight=1.7817,

                          n_estimators=2200,reg_alpha=0.4640,reg_lambda=0.8571,subsample=0.5213,silent=1,nthread=-1)
score=rmsle_cv(lasso)

print('\nlasso score: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

score=rmsle_cv(ENet)

print('\nENet score: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

score=rmsle_cv(KRR)

print('\nKRR score: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

score=rmsle_cv(GBoost)

print('\nGBoost score: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

score=rmsle_cv(model_xgb)

print('\nmodel_xgb score: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))