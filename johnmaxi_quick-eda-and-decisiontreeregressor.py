# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read in the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.head())
numerics = ['int32', 'int64', 'float32', 'float64']



numeric_train = train.select_dtypes(include=numerics)

print(numeric_train.head())
numeric_train.plot(x='LotArea', y='SalePrice', kind='scatter')
numeric_train.plot(x='MSSubClass', y='SalePrice', kind='scatter')
numeric_train.plot(x='LotFrontage', y='SalePrice', kind='scatter')
numeric_train.plot(x='OverallQual', y='SalePrice', kind='scatter')
numeric_train.plot(x='OverallCond', y='SalePrice', kind='scatter')
numeric_train['OverallQualCond'] = numeric_train.OverallQual*numeric_train.OverallCond

numeric_train.plot(x='OverallQualCond', y='SalePrice', kind='scatter')
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt



reg = DecisionTreeRegressor()

quality = train.OverallQual.reshape(-1,1)

reg.fit(quality, train.SalePrice)

plt.plot(quality, reg.predict(quality), color='red', linewidth=1)

plt.scatter(quality, train.SalePrice, alpha=0.5, c=train.SalePrice)

plt.xlabel = 'Overall Quality'

plt.ylabel = 'Sale Price'

plt.show()
from sklearn.linear_model import LinearRegression



reg = LinearRegression()

quality = train.OverallQual.reshape(-1,1)

reg.fit(quality, train.SalePrice)

plt.plot(quality, reg.predict(quality), color='red', linewidth=1)

plt.scatter(quality, train.SalePrice, alpha=0.5, c=train.SalePrice)

plt.xlabel = 'Overall Quality'

plt.ylabel = 'Sale Price'

plt.show()
from sklearn.model_selection import train_test_split



quality = train.OverallQual.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(quality, train.SalePrice, test_size = .3, random_state = 0)

reg = DecisionTreeRegressor()



reg.fit(X_train, y_train)

print('Simple DecisionTreeRegressor r^2: {}'.format(reg.score(X_test, y_test)))
X_train, X_test, y_train, y_test = train_test_split(quality, train.SalePrice, test_size = .3, random_state = 0)

reg_lin = LinearRegression()



reg_lin.fit(X_train, y_train)

print('Simple LinearRegreesion r^2: {}'.format(reg_lin.score(X_test, y_test)))