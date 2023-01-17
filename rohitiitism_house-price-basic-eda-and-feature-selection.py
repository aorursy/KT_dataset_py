# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train =pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

pd.set_option('display.max_columns', None)

df_train.head()
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])

plt.xticks(rotation=90)

plt.show()
sns.scatterplot(df_train['GrLivArea'],df_train['SalePrice'])

plt.show()
sns.scatterplot(df_train['TotalBsmtSF'],df_train['SalePrice'])

plt.show()
sns.scatterplot(df_train['GarageArea'],df_train['SalePrice'])

plt.show()
sns.lineplot(df_train['OverallQual'],df_train['SalePrice'])

plt.show()
sns.lineplot(df_train['OverallCond'],df_train['SalePrice'])

plt.show()
sns.lineplot(df_train['YearBuilt'],df_train['SalePrice'])

plt.show()
df_train['Total Area']=df_train['GrLivArea']+df_train['TotalBsmtSF']+df_train['GarageArea']

df_train['Age']=df_train['YrSold']-df_train['YearRemodAdd']

df_train.head()
sns.scatterplot(df_train['Total Area'],df_train['SalePrice'])

plt.show()
sns.lineplot(df_train['Age'],df_train['SalePrice'])

plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_train=df_train.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'])
corr=df_train.corr()

plt.figure(figsize=(50,50))

sns.heatmap(corr,annot=True,cmap='coolwarm')

plt.show()











col=['OverallQual','Age','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea','Total Area','SalePrice']

sns.pairplot(df_train[col],height=3)

plt.show()
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')

df_test['Total Area']=df_test['GrLivArea']+df_test['TotalBsmtSF']+df_test['GarageArea']

df_test['Age']=df_test['YrSold']-df_test['YearRemodAdd']

cols=['OverallQual','Age','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea','Total Area']

df_test=df_test[cols]

df_test.head()
col=['OverallQual','Age','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea','Total Area','SalePrice']

df_train=df_train[col]

df_train.head()
X=df_train.drop(columns='SalePrice')

y=df_train['SalePrice']
y=y.values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.ensemble import RandomForestRegressor

ran_for_reg=RandomForestRegressor(n_estimators=1000,random_state=0)

ran_for_reg.fit(X_train,y_train)

y_pred = ran_for_reg.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error

r2=r2_score(y_test,y_pred)

mse=mean_squared_error(y_test,y_pred)
print(r2)

print(mse)
df_test[cols] = df_test[cols].fillna(0)

X_final=df_test

y_final=ran_for_reg.predict(X_final)

y_final
submission=pd.DataFrame()

submission['Id']=df_test.index




submission['SalePrice'] = y_final

submission.head()
submission.to_csv('submission.csv')