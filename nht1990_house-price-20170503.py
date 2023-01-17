# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

train.head()
train.describe()
train.columns
plt.plot(train['GrLivArea'],train['SalePrice'],'o')
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
predictor=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF']

train[predictor].describe()
k = 10  

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[list(cols)].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},yticklabels=cols.values, xticklabels=cols.values)

plt.show()

print (cols)
from sklearn.linear_model import LinearRegression

predictors = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x_train=train[predictors]

y_train=train['SalePrice']

linreg = LinearRegression()

linreg.fit(x_train,y_train)

round(linreg.score(x_train, y_train) * 100, 2)
test=pd.read_csv('../input/test.csv')

test['GarageCars']=test['GarageCars'].fillna(train['GarageCars'].median())

test['GarageArea']=test['GarageArea'].fillna(train['GarageArea'].median())

test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(train['TotalBsmtSF'].median())

x_test.describe()
submission=linreg.predict(x_test)