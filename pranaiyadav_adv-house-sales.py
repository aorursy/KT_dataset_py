# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

dff = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.shape

df.head()
df.shape, dff.shape
df.columns
[ col for col in df.columns if df[col].isnull().sum()>0]
[ col for col in dff.columns if dff[col].isnull().sum()>0]
for i in df.columns:

    if df[i].isna().sum()>0:

        print(i,df[i].isna().sum())
for i in dff.columns:

    if dff[i].isna().sum()>0:

        print(i,dff[i].isna().sum())
df.info()
dff.info()
df = df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)

dff = dff.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)
df.LotFrontage.value_counts()
df.LotFrontage.isna().sum()
for i in df.LotFrontage[df.LotFrontage.isna()==True].index:

    df.LotFrontage[i] = np.random.randint(50,80)
for i in dff.LotFrontage[dff.LotFrontage.isna()==True].index:

    dff.LotFrontage[i] = np.random.randint(50,80)
df.MasVnrType.value_counts()
dff.MasVnrType.value_counts()
df.MasVnrType.fillna('None',inplace=True)

dff.MasVnrType.fillna('None',inplace=True)
df.MasVnrArea.value_counts()
dff.MasVnrArea.value_counts()
df.MasVnrArea.fillna(0,inplace=True)

dff.MasVnrArea.fillna(0,inplace=True)
df.BsmtCond.value_counts()
dff.BsmtCond.value_counts()
df.BsmtCond.fillna('TA',inplace=True)

dff.BsmtCond.fillna('TA',inplace=True)
df.BsmtQual.value_counts()
dff.BsmtQual.value_counts()
c=0

for i in df[df.BsmtQual.isna()==True].index:

    c=c+1

    if c%2==0:

        df.BsmtQual[i] = 'TA'

    else:

        df.BsmtQual[i] = 'Gd'
c=0

for i in dff[dff.BsmtQual.isna()==True].index:

    c=c+1

    if c%2==0:

        dff.BsmtQual[i] = 'TA'

    else:

        dff.BsmtQual[i] = 'Gd'
df.BsmtExposure.value_counts()
dff.BsmtExposure.value_counts()
df.BsmtExposure.fillna('No',inplace=True)

dff.BsmtExposure.fillna('No',inplace=True)
df.BsmtFinType1.value_counts()
dff.BsmtFinType1.value_counts()
c=0

for i in df[df.BsmtFinType1.isna()==True].index:

    c=c+1

    if c%2==0:

        df.BsmtFinType1[i] = 'Unf'

    else:

        df.BsmtFinType1[i] = 'GLQ'
c=0

for i in dff[dff.BsmtFinType1.isna()==True].index:

    c=c+1

    if c%2==0:

        dff.BsmtFinType1[i] = 'Unf'

    else:

        dff.BsmtFinType1[i] = 'GLQ'
df.BsmtFinType2.value_counts()
dff.BsmtFinType2.value_counts()
for i in df[df.BsmtFinType2.isna()==True].index:

        df.BsmtFinType2[i] = 'Unf'
for i in dff[dff.BsmtFinType2.isna()==True].index:

        dff.BsmtFinType2[i] = 'Unf'
df.Electrical.value_counts()
dff.Electrical.value_counts()
for i in df[df.Electrical.isna()==True].index:

        df.Electrical[i] = 'SBrkr'
for i in dff[dff.Electrical.isna()==True].index:

        dff.Electrical[i] = 'SBrkr'
df.GarageCond.value_counts()
dff.GarageCond.value_counts()
for i in df[df.GarageCond.isna()==True].index:

        df.GarageCond[i] = 'TA'
for i in dff[dff.GarageCond.isna()==True].index:

        dff.GarageCond[i] = 'TA'
df.GarageQual.value_counts()
dff.GarageQual.value_counts()
for i in df[df.GarageQual.isna()==True].index:

        df.GarageQual[i] = 'TA'
for i in dff[dff.GarageQual.isna()==True].index:

        dff.GarageQual[i] = 'TA'
df.GarageType.value_counts()
dff.GarageType.value_counts()
for i in df[df.GarageType.isna()==True].index:

        df.GarageType[i] = 'Attchd'
for i in dff[dff.GarageType.isna()==True].index:

        dff.GarageType[i] = 'Attchd'
df.GarageYrBlt.value_counts().head(10)
dff.GarageYrBlt.value_counts().head(10)
for i in df[df.GarageYrBlt.isna()==True].index:

        df.GarageYrBlt[i] = np.random.randint(2003,2008)
for i in dff[dff.GarageYrBlt.isna()==True].index:

        dff.GarageYrBlt[i] = np.random.randint(2003,2008)
df.GarageFinish.value_counts()
dff.GarageFinish.value_counts()
c=0

for i in df[df.GarageFinish.isna()==True].index:

    c=c+1

    if c%2==0:

        df.GarageFinish[i] = 'Unf'

    elif c%3==0:

        df.GarageFinish[i]='RFn'

    else:

        df.GarageFinish[i]='Fin'
c=0

for i in dff[dff.GarageFinish.isna()==True].index:

    c=c+1

    if c%2==0:

        dff.GarageFinish[i] = 'Unf'

    elif c%3==0:

        dff.GarageFinish[i]='RFn'

    else:

        dff.GarageFinish[i]='Fin'
[ col for col in df.columns if df[col].isnull().sum()>0]
[ col for col in dff.columns if dff[col].isnull().sum()>0]
dff.MSZoning.value_counts()
df.MSZoning.value_counts()
for i in dff[dff.MSZoning.isna()==True].index:

        dff.MSZoning[i] = 'RL'
dff.BsmtFinSF1.value_counts()
df.BsmtFinSF1.value_counts()
for i in dff[dff.BsmtFinSF1.isna()==True].index:

        dff.BsmtFinSF1[i] = 0
dff.BsmtFinSF1 = dff.BsmtFinSF1.astype('int64')
df.BsmtFinSF2.value_counts()
for i in dff[dff.BsmtFinSF2.isna()==True].index:

        dff.BsmtFinSF2[i] = 0
dff.BsmtFinSF2 = dff.BsmtFinSF2.astype('int64')
dff.BsmtUnfSF.value_counts()
df.BsmtUnfSF.value_counts()
dff.BsmtUnfSF = dff.BsmtUnfSF.astype('int64')
for i in dff[dff.BsmtUnfSF.isna()==True].index:

        dff.BsmtUnfSF[i] = 0
dff.TotalBsmtSF.value_counts()
df.TotalBsmtSF.value_counts()
c=0

for i in dff[dff.TotalBsmtSF.isna()==True].index:

    c=c+1

    if c%2==0:

        dff.TotalBsmtSF[i] = 0

    else:

        dff.TotalBsmtSF[i]=864
dff.TotalBsmtSF = df.TotalBsmtSF.astype('int64')
dff.BsmtFullBath.value_counts()
df.BsmtFullBath.value_counts()
for i in dff[dff.BsmtFullBath.isna()==True].index:

        dff.BsmtFullBath[i] = 0
dff.BsmtFullBath = dff.BsmtFullBath.astype('int64')
dff.BsmtHalfBath.value_counts()
df.BsmtHalfBath.value_counts()
for i in dff[dff.BsmtHalfBath.isna()==True].index:

        dff.BsmtHalfBath[i] = 0
dff.BsmtHalfBath = dff.BsmtHalfBath.astype('int64')
dff.KitchenQual.value_counts()
df.KitchenQual.value_counts()
c=0

for i in dff[dff.KitchenQual.isna()==True].index:

    c=c+1

    if c%2==0:

        dff.KitchenQual[i] = 'TA'

    else:

        dff.KitchenQual[i]='Gd'
dff.Functional.value_counts()
df.Functional.value_counts()
for i in dff[dff.Functional.isna()==True].index:

        dff.Functional[i] = 'Typ'
dff.GarageCars.value_counts()
df.GarageCars.value_counts()
for i in dff[dff.GarageCars.isna()==True].index:

        dff.GarageCars[i] = 2
dff.GarageCars = dff.GarageCars.astype('int64')
dff.SaleType.value_counts()
df.SaleType.value_counts()
for i in dff[dff.SaleType.isna()==True].index:

        dff.SaleType[i] = 'WD'
dff.GarageArea.value_counts()
df.GarageArea.value_counts()
for i in dff[dff.GarageArea.isna()==True].index:

        dff.GarageArea[i] = 0
dff.GarageArea = dff.GarageArea.astype('int64')
[col for col in dff.columns if dff[col].isna().sum()>0]
df.info()
dff.info()
dff.Exterior1st.value_counts()
df.Exterior1st.value_counts()
for i in dff[dff.Exterior1st.isna()==True].index:

        dff.Exterior1st[i] = 'VinylSd'
for i in dff[dff.Exterior2nd.isna()==True].index:

        dff.Exterior2nd[i] = 'VinylSd'
df.drop('Utilities',axis=1,inplace=True)

dff.drop('Utilities',axis=1,inplace=True)
df.iloc[:5,:20]
x = df.describe(include='all')
df.info()
x.iloc[:,60:]
df.columns
df.drop(['Street','LandContour','LandSlope','Condition1','Condition2','BldgType','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir','Electrical','GarageQual','GarageCond','PavedDrive','SaleType'],axis=1,inplace=True)

dff.drop(['Street','LandContour','LandSlope','Condition1','Condition2','BldgType','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir','Electrical','GarageQual','GarageCond','PavedDrive','SaleType'],axis=1,inplace=True)
df = pd.get_dummies(df,drop_first=True)

dff = pd.get_dummies(dff,drop_first=True)
df.shape, dff.shape
l1 = df.columns

l2 = dff.columns
l1 = set(l1)

l2 = set(l2)
l1 - l2
df.drop(['Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','HouseStyle_2.5Fin'],axis=1,inplace=True)
df.shape, dff.shape
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
x = df.corr()['SalePrice']
x = x.reset_index()
x.iloc[:20,1:]
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression

lr  = LinearRegression(normalize=True)
X = df.drop('SalePrice',axis=1)

y = df.SalePrice
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error
X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0,test_size=0.2)
lr.fit(X_train,y_train)

lr.score(X_train,y_train)
lr.score(X_valid,y_valid)
from sklearn.ensemble import RandomForestRegressor

rr = RandomForestRegressor(n_estimators=100,n_jobs=4,random_state=0).fit(X_train,y_train)
rr.score(X_train,y_train), rr.score(X_valid,y_valid)
from xgboost import XGBRegressor

dd = XGBRegressor(n_estimators=1000,n_jobs=4,learning_rate=0.05).fit(X_train,y_train)
dd.score(X_train,y_train), dd.score(X_valid,y_valid)
ff = X.assign(const=1)

vif = pd.DataFrame([variance_inflation_factor(ff.values,i) for i in range(ff.shape[1])],index=ff.columns)

vif
vif.reset_index(inplace=True)

vif.columns = ['col','val']

vif.head()
vif.sort_values(by='val',inplace=True)
v = vif[vif.val<=10].col
len(v)
X1 = df[v]

X_train1,X_valid1,y_train1,y_valid1 = train_test_split(X1,y,test_size=0.2)
from xgboost import XGBRegressor

dd1 = XGBRegressor(n_estimators=1000,n_jobs=4).fit(X_train1,y_train1)
dd1.score(X_train1,y_train1), dd1.score(X_valid1,y_valid1)
rr1 = RandomForestRegressor(n_estimators=1500,random_state=0).fit(X_train1,y_train1)

rr1.score(X_train1,y_train1), rr1.score(X_valid1,y_valid1)
lr1 = LinearRegression().fit(X_train1,y_train1)

lr1.score(X_train1,y_train1), lr1.score(X_valid1,y_valid1)
dff = dff[v]

pre = dd1.predict(dff)
pre.shape
dff.Id = dff.Id.astype(int)
Submission=pd.DataFrame( { 'Id' : dff['Id'] , 'SalePrice' : pre} )

Submission.to_csv('Submission.csv',index=False)