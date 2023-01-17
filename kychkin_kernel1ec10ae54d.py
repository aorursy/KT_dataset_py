# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OrdinalEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
columnz = []
for col in train.columns:
    if str(train[col].dtype)=='object':
        columnz.append(col)

train[columnz] = train[columnz].fillna('nana',inplace = True)
test[columnz] = test[columnz].fillna('nana',inplace = True)
train = train.fillna(-1)
test = test.fillna(-1)

enCoder = OrdinalEncoder()
enCoder.fit(pd.concat([train[columnz],test[columnz]]))

train[columnz] = enCoder.transform(train[columnz]).astype(int)
test[columnz] = enCoder.transform(test[columnz]).astype(int)
#columnBig = []
#for col in train.columns:
#    maximum = train[col].max()
#    if (maximum>100)and(col!='Id')and(col!='SalePrice'):   
#        print(col,maximum)
#        columnBig.append(col)
#train[columnBig]
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
lasso_regression = Lasso(alpha=0.1)
lasso_regression.fit(train.filter(regex='[^SalePrice]') , train['SalePrice'])
#print(lasso_regression.predict(train.filter(regex='[^SalePrice]')))
#print(train.filter(regex='SalePrice'))
#print(lasso_regression.predict(train.filter(regex='[^SalePrice]'))[0])
#print(train['SalePrice'][0])
print('Id,SalePrice')
for row in range(1459):
    print(row, ',',lasso_regression.predict(test)[row], sep='')