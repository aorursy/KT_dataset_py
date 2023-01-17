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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/home-data-for-ml-course/train.csv')

test=pd.read_csv('../input/home-data-for-ml-course/test.csv')

train.head()
train.info()
train.describe()
plt.figure(figsize=(20,6))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
nullValue=['PoolQC','Fence','MiscFeature','Alley']

train.drop(nullValue,axis=1,inplace=True)

test.drop(nullValue,axis=1,inplace=True)
train['LotFrontage']=train['LotFrontage'].fillna((train['LotFrontage']).mean())

train['GarageYrBlt']=train['GarageYrBlt'].fillna((train['GarageYrBlt']).mean())

train['MasVnrArea']=train['MasVnrArea'].fillna((train['MasVnrArea']).mean())



test['LotFrontage']=test['LotFrontage'].fillna((test['LotFrontage']).mean())

test['GarageYrBlt']=test['GarageYrBlt'].fillna((test['GarageYrBlt']).mean())

test['MasVnrArea']=test['MasVnrArea'].fillna((test['MasVnrArea']).mean())

plt.figure(figsize=(20,6))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
na=['Electrical','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType']

def impute_cate(cols):

    age=cols[0]

    if pd.isnull(age):

        return "Missing"

    else:

        return cols
for i in range(len(na)):

    train[na[i]]=train[[na[i]]].apply(impute_cate,axis=1)

    test[na[i]]=test[[na[i]]].apply(impute_cate,axis=1)
train.dropna(inplace=True)

plt.figure(figsize=(20,6))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
test.dropna(inplace=True)

plt.figure(figsize=(20,6))

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap="viridis")
train.plot(kind='box',figsize=(38,20),subplots=True,layout=(8,8))

plt.show()
sns.boxplot(x='LotArea',data=train)
outNan=['ScreenPorch','PoolArea','MiscVal','3SsnPorch','EnclosedPorch','KitchenAbvGr','BsmtHalfBath','LowQualFinSF','BsmtFinSF2']

train.drop(outNan,axis=1,inplace=True)

test.drop(outNan,axis=1,inplace=True)
train.info()
train.plot(kind='box',figsize=(38,20),subplots=True,layout=(4,8))

plt.show()
out=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','MasVnrArea','BsmtFinSF1',

    'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','BsmtFullBath','BedroomAbvGr',

    'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF']
sns.scatterplot(x='LotArea',y='SalePrice',data=train)
sns.scatterplot(x='LotFrontage',y='SalePrice',data=train)
def outliers(var):

    a=[]

    q1=train[var].quantile(0.25)

    q2=train[var].quantile(0.75)

    iqr=q2-q1

    

    ulim=float(q2+(1.5*iqr))

    llim=float(q1-(1.5*iqr))

    

    for i in train[var]:

        if i>ulim:

            i=np.nan

        elif i<llim:

            i=np.nan

        else:

            i=i

        a.append(i)

    return a
for col in train.select_dtypes(exclude='object').columns:

    train[col] = outliers(col)
train.plot(kind='box',figsize=(38,20),subplots=True,layout=(4,8))

plt.show()
train.isna().sum()
for i in train.select_dtypes(exclude='object').columns:

    train[i]=train[i].fillna(train[i].mean())
train.isna().sum()
train.info()
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
train.drop('FireplaceQu',inplace=True,axis=1)

plt.figure(figsize=(20,6))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
na1=['Electrical','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType']

encoded=train[na1].apply(encoder.fit_transform)

encoded.head(5)
train.drop(na1,inplace=True,axis=1)
train.info()
train=pd.concat([train,encoded],axis=1)

train.info()


X=train.drop(['SalePrice'],axis=1)

y=train['SalePrice']
hot=pd.get_dummies(X,drop_first=True)

hot.shape
hot.info()
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test=train_test_split(hot,y,test_size=0.33,random_state=101)

model = RandomForestRegressor(n_estimators=100)

model.fit(X_train,y_train)
pred=model.predict(X_test)
from sklearn.metrics import mean_squared_log_error
print('RMSE:', np.sqrt(mean_squared_log_error(y_test, pred)))
from sklearn.metrics import classification_report,confusion_matrix,r2_score,explained_variance_score
print(r2_score(y_test,pred))
print(explained_variance_score(y_test,pred))
sns.distplot((y_test-pred),bins=50)
plt.scatter(y_test,pred)
modelLinear = LinearRegression()

modelLinear.fit(X_train,y_train)

predictLinear=model.predict(X_test)

print('RMSE:', np.sqrt(mean_squared_log_error(y_test, predictLinear)))
