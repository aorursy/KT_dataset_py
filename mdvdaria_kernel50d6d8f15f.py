# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/test.csv')
train.shape
test.shape
train.head()
x_train = train[['LotArea','LotFrontage']].copy()
y_train = train['SalePrice'].copy()
x_test = test[['LotArea','LotFrontage']].copy()
y_train.head()
x_train.head()
x_train.shape
x_train.isnull().sum()
x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
x_scaler =  StandardScaler()
y_scaler =  StandardScaler()
x_train_scaler = x_scaler.fit_transform(x_train)
x_test_scaler = x_scaler.transform(x_test)
lr = LinearRegression()
lr.fit(x_train_scaler,y_train)
pred = lr.predict(x_test_scaler)
pd.read_csv('../input/sample_sumbission.csv').head()
sub = pd.DataFrame(data = {'Id' : test.Id, 'SalePrice' :pred})
sub.to_csv('submission.csv',index = False ) 
