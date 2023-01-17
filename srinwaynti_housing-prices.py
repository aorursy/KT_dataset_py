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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from fancyimpute import KNN

from sklearn.preprocessing import StandardScaler
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train_shape=train.shape

print(train_shape)

train.head()

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_shape=test.shape

print(test_shape)

test.head()
test_Id=test['Id']

train_Id=train['Id']
full_data=pd.concat([train.drop('SalePrice',axis=1),test])

full_data.shape


full_data.info()
plt.figure(figsize=(10,8))

sns.heatmap(full_data.isnull(),cbar=False)

full_data['Electrical'].describe()
full_data['Electrical']=full_data['Electrical'].fillna('SBrkr')

full_data['Electrical'].isnull().sum()
cols=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']

for each in cols:

    full_data[each]=full_data[each].fillna(full_data[each].mean())

full_data[1460:].head()
full_data['MasVnrType'].unique()
full_data['MasVnrType'].describe()

full_data['MasVnrType']=full_data['MasVnrType'].fillna('None')
full_data['MasVnrArea'].describe()
full_data['MasVnrArea'].isna().sum()
sns.boxplot(full_data['MasVnrArea'])
cols=[]

for each in full_data.columns:

    if full_data[each].dtype!=object:

        cols.append(each)

    else:

        pass

        

data1=full_data[cols]

data1.isnull().sum()

data1=data1.drop('Id',axis=1)

#MasVnrArea_impute=data1.drop(['LotFrontage','GarageYrBlt'],axis=1)
data1[1460:].head()
data1_cols=list(data1)

data1 = pd.DataFrame(KNN(k=5).fit_transform(data1))

data1.columns = data1_cols
data1.isnull().sum()
data1.info()
data1.head(10)
data1[1460:].head()
for each in data1_cols:

    full_data[each]=data1[each]
full_data.info()
full_data[1460:].head()
full_data['SaleType'].describe()

full_data['SaleType']=full_data['SaleType'].fillna('WD')
full_data=full_data.drop(['PoolQC','MiscFeature','Fence','Alley'],axis=1)
full_data['FireplaceQu']=full_data['FireplaceQu'].fillna('Gd')
full_data['BsmtCond']=full_data['BsmtCond'].fillna('TA')
full_data['MSZoning']=full_data['MSZoning'].fillna('RL')
full_data['Utilities']=full_data['Utilities'].fillna('AllPub')
full_data['Utilities'].isna().sum()


full_data['Exterior1st']=full_data['Exterior1st'].fillna('VinylSd')

full_data['Exterior2nd']=full_data['Exterior2nd'].fillna('VinylSd')

full_data['BsmtQual']=full_data['BsmtQual'].fillna('TA')

full_data['BsmtExposure']=full_data['BsmtExposure'].fillna('No')

full_data['BsmtFinType1']=full_data['BsmtFinType1'].fillna('Unf')

full_data['BsmtFinType2']=full_data['BsmtFinType2'].fillna('Unf')

full_data['KitchenQual']=full_data['KitchenQual'].fillna('TA')

full_data['Functional']=full_data['Functional'].fillna('Typ')

full_data['GarageType']=full_data['GarageType'].fillna('Attchd')

full_data['GarageFinish']=full_data['GarageFinish'].fillna('Unf')

full_data['GarageQual']=full_data['GarageQual'].fillna('TA')

full_data['GarageCond']=full_data['GarageCond'].fillna('TA')
full_data[1460:].head()
sns.heatmap(full_data.isnull(),cbar=False)
plt.figure(figsize=(30,20))

sns.heatmap(full_data.drop('Id',axis=1).corr(),annot=True)
full_data=full_data.drop(['YearBuilt','TotRmsAbvGrd','YearRemodAdd','GarageYrBlt','GarageArea'],axis=1)
full_data[1460:].head()
full_data.drop('Id',axis=1,inplace=True)

#full_data=pd.get_dummies(full_data)
full_data['MSSubClass'].unique()

full_data['MSSubClass']=full_data['MSSubClass'].astype('str')

full_data['OverallQual']=full_data['OverallQual'].astype('str')

full_data['OverallCond']=full_data['OverallCond'].astype('str')
full_data.info()
full_data[1460:].head()
full_data_1=pd.get_dummies(full_data)
full_data_1.shape
train1=full_data_1[:1460]

test1=full_data_1[1460:]
scalar=StandardScaler()

train1=scalar.fit_transform(train1)

test1=scalar.transform(test1)
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
train_y=train['SalePrice']
model=GaussianNB()

model1=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=100,

                      max_features='auto', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=100,

                      n_jobs=None, oob_score=False, random_state=None,

                      verbose=0, warm_start=False)

model2=LinearRegression()
model.fit(train1,train_y)
model1.fit(train1,train_y)
model2.fit(train1,train_y)
predict_test=model.predict(test1)
predict_test1=model1.predict(test1)
predict_test2=model2.predict(test1)
my_submission=pd.DataFrame({'Id':test_Id,'SalePrice':predict_test})

my_submission1=pd.DataFrame({'Id':test_Id,'SalePrice':predict_test1})

my_submission2=pd.DataFrame({'Id':test_Id,'SalePrice':predict_test2})
my_submission.to_csv('submission.csv',index=False)

my_submission1.to_csv('submission1.csv',index=False)

my_submission2.to_csv('submission2.csv',index=False)