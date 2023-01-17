import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



sns.set()
tips = sns.load_dataset("tips")
df = tips.copy()

print(df.head())
sns.distplot(df['total_bill'],kde=False, bins = 20)
sns.jointplot(x = 'total_bill',y = 'tip', data = df)
sns.pairplot(df,hue = 'sex')
sns.rugplot(df['total_bill'])
sns.barplot(x = 'sex', y = 'total_bill', data = df, estimator = np.std)
sns.countplot(x = 'sex', data = df)
sns.boxplot(x = 'day',y = 'total_bill', data = df, hue = 'smoker')
sns.violinplot(x = 'day',y = 'total_bill', data = df, hue = 'sex',split = True, palette = 'seismic')
sns.stripplot(x = 'day', y = 'total_bill', data = df)
sns.swarmplot(x = 'day', y = 'total_bill', data = df)
sns.factorplot(x = 'day', y = 'total_bill', data = df, kind = 'violin');
flights = sns.load_dataset('flights')
flights.head()
tips.corr()
sns.heatmap(tips.corr(),annot=True,cmap = 'coolwarm')
fp = flights.pivot_table(index = 'month', columns = 'year', values = 'passengers')
sns.heatmap(fp,cmap = 'coolwarm', linecolor = 'white', linewidths = .5)
sns.clustermap(fp, cmap = 'coolwarm', standard_scale=1)
iris = sns.load_dataset('iris')

iris.head()
iris['species'].unique()
g = sns.PairGrid(iris)

g.map_upper(plt.scatter)

g.map_diag(sns.distplot)

g.map_lower(sns.kdeplot)
k = sns.FacetGrid(data = tips, col = 'time', row = 'smoker')

k.map(sns.distplot, 'total_bill')
k = sns.FacetGrid(data = tips, col = 'time', row = 'smoker')

k.map(plt.scatter, 'total_bill', 'tip')
sns.lmplot(x = 'total_bill', y = 'tip', data = tips, col = 'sex', row = 'time')
sns.lmplot(x = 'total_bill',y = 'tip',hue = 'sex', data = tips, palette = 'seismic')