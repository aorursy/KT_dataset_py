import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

sns.set(color_codes=True)

titanic=pd.read_csv('../input/train.csv')

titanic.head()
sns.distplot(titanic['Fare'])
sns.distplot(titanic['Fare'],bins=100)
sns.distplot(titanic['Fare'],hist=False, rug=True)

titanic.head()
sns.jointplot(x="Fare", y="Age", data=titanic);
sns.jointplot(x="Fare", y="Age",kind='hex', data=titanic);
sns.jointplot(x="Fare", y="Age",kind='kde', data=titanic);
sns.pairplot(titanic)
g = sns.PairGrid(titanic)

g.map_diag(sns.barplot)

g.map_offdiag(sns.jointplot,kind='kde')