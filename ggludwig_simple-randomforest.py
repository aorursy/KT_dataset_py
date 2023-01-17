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
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")

test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")

stores = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
#preparing train df

train=train.merge(stores,how='left',on='Store') #merge with stores to get more features

train.replace({'A': 1, 'B': 2,'C':3},inplace=True) #

train['Date']=pd.to_datetime(train['Date'])

train['dia']=pd.to_datetime(train['Date']).dt.day

train['mes']=pd.to_datetime(train['Date']).dt.month

train['ano']=pd.to_datetime(train['Date']).dt.year

train.head()
test=test.merge(stores,how='left',on='Store')

test.replace({'A': 1, 'B': 2,'C':3},inplace=True)

test['Date']=pd.to_datetime(test['Date'])

test['dia']=pd.to_datetime(test['Date']).dt.day

test['mes']=pd.to_datetime(test['Date']).dt.month

test['ano']=pd.to_datetime(test['Date']).dt.year

test.head()
X = train.loc[:, train.columns != 'Weekly_Sales'].drop('Date',axis=1)

y = train.loc[:, train.columns == 'Weekly_Sales']

rf = RandomForestRegressor()

rf.fit(X, y)

test['Weekly_Sales'] = rf.predict(test.drop('Date',axis=1))
test['ID']=test['Store'].astype(str)+'_'+test['Dept'].astype(str)+'_'+test['Date'].astype(str)

test[['ID','Weekly_Sales']].to_csv('sub.csv',index=False)