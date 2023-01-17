import pandas



dataset = pandas.read_csv('../input/train.csv')

dataset.head()
dataset.columns
dataset.SalePrice.describe()
dataset.corr(method="spearman").sort_values(by='SalePrice', ascending=False)['SalePrice']
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.set(style="ticks")

sns.factorplot(x='OverallQual', y="SalePrice", data=dataset)
sns.factorplot(x='GarageCars', y="SalePrice", data=dataset)
sns.lmplot(x='GrLivArea', y="SalePrice", data=dataset)
sns.lmplot(x='YearBuilt', y="SalePrice", data=dataset)