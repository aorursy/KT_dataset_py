

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.describe()



train.head()
train.info()
from sklearn.linear_model import LinearRegression

X = train['GrLivArea'].values[:,np.newaxis]

y = train['SalePrice'].values

model = LinearRegression()

model.fit(X,y)

plt.scatter(X,y)

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')