import pandas as pd

train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
train.head()
train.shape
print(train.dtypes.unique())
len(train.select_dtypes(include=['O']).columns)
len(train.select_dtypes(include=['int64']).columns)
len(train.select_dtypes(include=['float64']).columns)
pd.set_option('precision', 3)

train.describe()
y=train.count()[train.count()!=1460.000]
y.shape
sns.barplot(y.index,y.values)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(train.corr(method='pearson'))
y=abs(train.skew())

plt.figure(figsize=(7,7))

sns.barplot(y.index,y.values)

plt.xticks(rotation=90)

plt.show()
train.hist(figsize=(20,20))

plt.show()
x=train.select_dtypes(exclude='object')

f=train.describe().columns

y=train.count()[train.count()==1460].index

g=f.intersection(y)

for feat in g:

  sns.distplot(train[feat])

  plt.show()

g=train.select_dtypes(exclude='object').columns

for x in g:

  sns.boxplot(train[x],orient='v')

  plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(train.corr(method='pearson'))
# since scatter_matrix takes a lot of time taking less features for demo

g=['SalePrice','LotArea','OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','TotalBsmtSF','GarageCars','OverallCond','KitchenAbvGr']

sns.pairplot(train[g])