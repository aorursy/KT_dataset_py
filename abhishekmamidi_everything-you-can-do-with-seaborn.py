import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
tips = sns.load_dataset('tips')
tips.head()
sns.barplot(x='day', y='tip', data=tips)
sns.barplot(x='day', y='total_bill', data=tips, hue='sex', palette='winter_r')
sns.barplot(x='day', y='total_bill', data=tips, hue='smoker', palette='winter_r')
sns.barplot(x='total_bill', y='day', data=tips, palette='spring')
sns.barplot(x='day', y='tip', data=tips, palette='spring', order=['Sat', 'Sun', 'Thur', 'Fri'])
from numpy import median
sns.barplot(x='day', y='total_bill', data=tips, estimator=median, palette='spring')
sns.barplot(x='smoker', y='tip', data=tips, estimator=median, hue='sex', palette='coolwarm')
sns.barplot(x='smoker', y='tip', data=tips, ci=100)
# ci - confience interval (error part)
sns.barplot(x='day', y='total_bill', data=tips, capsize=0.3, palette='husl')
sns.barplot(x='day', y='total_bill', data=tips, hue='sex', capsize=0.2, palette='husl')
tips.head()
sns.barplot(x='size', y='tip', data=tips, capsize=0.15, palette='autumn')
sns.barplot(x='size', y='tip', data=tips, capsize=0.15, palette='husl')
sns.barplot(x='size', y='tip', data=tips, capsize=0.15, color='red', saturation=0.7)
num = np.random.randn(100)
sns.distplot(num)
sns.distplot(num, color='red')
label_dist = pd.Series(num, name='variable x')
sns.distplot(label_dist)
sns.distplot(label_dist, vertical=True, color='red')
# Univariate Kernel Density Estimate(KDE) plot.
sns.distplot(label_dist, hist=False)
sns.distplot(label_dist, rug=True, hist=False, color='green')
tips = sns.load_dataset('tips')
tips.head()
sns.boxplot(x=tips['size'])
sns.boxplot(x=tips['total_bill'])
tips['total_bill'].describe()
sns.boxplot(x='sex', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', hue='sex', data=tips, palette='husl')
tips.head()
sns.boxplot(x='day', y='total_bill', data=tips, hue='time')
sns.boxplot(x='day', y='total_bill', data=tips, order=['Sat', 'Sun', 'Thur', 'Fri'])

iris = sns.load_dataset('iris')
iris.head()
sns.boxplot(data=iris)
sns.distplot(iris.sepal_width)
sns.boxplot(data=iris, orient='horizontal', palette='husl')
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day', y='total_bill', data=tips, color='black')
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day', y='total_bill', data=tips, color='0.3')
tips=sns.load_dataset('tips')
tips.head()
sns.stripplot(x=tips['tip'], color='green')
sns.stripplot(x=tips['total_bill'], color='blue')
sns.boxplot(x='day', y='total_bill', data=tips)
sns.stripplot(x='day', y='total_bill', data=tips)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=0.2)
sns.stripplot(x='total_bill', y='day', data=tips, jitter=True)
sns.stripplot(x='total_bill', y='day', data=tips, linewidth=0.8, jitter=True)
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex', jitter=True)
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex', jitter=True, split=True)
sns.stripplot(x='day', y='total_bill', data=tips, hue='smoker', jitter=True, split=True)
sns.stripplot(x='sex', y='tip', hue='day', data=tips, marker='D', jitter=True)
sns.stripplot(x='sex', y='tip', hue='day', data=tips, marker='D', jitter=True, size=7)
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')
sns.stripplot(x='day', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
sns.violinplot(x='day', y='total_bill', data=tips, color='0.9')
iris = sns.load_dataset('iris')
x = sns.PairGrid(iris)
x = x.map(plt.scatter)
x = sns.PairGrid(iris)
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x = iris.petal_width.value_counts()
x = x.sort_index()
x.plot('bar')
x = sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
iris.species.value_counts()
x = sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x.add_legend()
x = sns.PairGrid(iris, hue='species', palette='winter_r') # coolwarm, husl, winter_r, RdBu.
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x.add_legend()
x = sns.PairGrid(iris, hue='species', palette='winter_r') # autumn, coolwarm, husl, winter_r, RdBu.
x = x.map_diag(plt.hist, histtype='step', linewidth=4)
x = x.map_offdiag(plt.scatter)
x.add_legend()
x = sns.PairGrid(iris, vars=['petal_length', 'petal_width'])
x = x.map(plt.scatter)
x = sns.PairGrid(iris, hue='species', vars=['petal_length', 'petal_width'])
x = x.map_diag(plt.hist, edgecolor='black')
x = x.map_offdiag(plt.scatter, edgecolor='black')
x = x.add_legend()
x = sns.PairGrid(iris, x_vars=['petal_length', 'petal_width'],
                y_vars=['sepal_length', 'sepal_width'])
x = x.map(plt.scatter)
x = sns.PairGrid(iris)
x = x.map_diag(plt.hist)
x = x.map_upper(plt.scatter)
x = x.map_lower(sns.kdeplot)
x = sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist, edgecolor='black')
x = x.map_upper(plt.scatter)
x = x.map_lower(sns.kdeplot)
x = x.add_legend()
x = sns.PairGrid(iris, hue='species', hue_kws={'marker': ['D', 's', '+']})
x = x.map(plt.scatter)
x = x.add_legend()
tips = sns.load_dataset('tips')
tips.head()
sns.violinplot(x=tips['tip'])
sns.violinplot(x='size', y='total_bill', data=tips)
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips)
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, split=True)
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, split=True, 
               inner='quartile')
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, split=True, 
               inner='quartile', scale='count')
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, 
               inner='quartile', scale='count')
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, 
               inner='quartile', scale='count')
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, 
               inner='stick', scale='count')
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, 
               inner='stick')
# Here, we can compare the number of customers on different days by width of violin plot.
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, 
               inner='stick', scale='count', scale_hue=False, split=True)
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, 
               inner='stick', scale='count', scale_hue=False, split=True, bw=0.7)
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, 
               inner='stick', scale='count', scale_hue=False, split=True, bw=0.1)
flights = sns.load_dataset('flights')
flights.head()
flights = flights.pivot('month', 'year', 'passengers')
flights
from matplotlib.colors import ListedColormap
sns.heatmap(flights)
sns.clustermap(flights)
sns.clustermap(flights, col_cluster=False)
sns.clustermap(flights, row_cluster=False)
sns.clustermap(flights, cmap='Blues_r', linewidth=1) # coolwarm, Blues_r
sns.clustermap(flights, cmap='coolwarm', linewidth=2, figsize=(8,6))
sns.clustermap(flights, cmap='coolwarm', standard_scale=1) # 1 - columns
sns.clustermap(flights, cmap='coolwarm', standard_scale=0) # 0 - rows
sns.clustermap(flights, cmap='coolwarm', z_score=0) # 0 - rows
normal = np.random.rand(12, 15)
sns.heatmap(normal, cmap='coolwarm')
sns.heatmap(normal, annot=True, cmap='coolwarm')
sns.heatmap(normal, vmin=0, vmax=2, cmap='coolwarm')
sns.heatmap(flights, cmap='coolwarm', annot=True, fmt='d', linewidths=0.3)
sns.heatmap(flights, cmap='coolwarm', annot=True, fmt='d', 
            linewidths=0.3, vmin=100, vmax=650)
sns.heatmap(flights, cmap='RdBu', annot=True, fmt='d') 
# color maps: RdBu, summer, coolwarm, winter_r
sns.heatmap(flights, center=flights.loc['June'][1954], annot=True, 
            fmt='d', cmap='coolwarm')
sns.heatmap(flights, center=flights.loc['March'][1959], annot=True, 
            fmt='d', cmap='coolwarm', cbar=False)
tips = sns.load_dataset('tips')
tips.head()
sns.FacetGrid(row='smoker', col='time', data=tips)
x = sns.FacetGrid(row='smoker', col='time', data=tips)
x = x.map(plt.hist, 'total_bill', edgecolor='black')
x = sns.FacetGrid(row='smoker', col='time', data=tips)
x = x.map(plt.hist, 'total_bill', edgecolor='black', color='green', 
          bins=15)
x = sns.FacetGrid(row='smoker', col='time', data=tips)
x = x.map(plt.scatter, 'total_bill', 'tip')
x = sns.FacetGrid(row='smoker', col='time', data=tips)
x = x.map(sns.regplot, 'total_bill', 'tip')
x = sns.FacetGrid(tips, col='time', hue='smoker')
x = x.map(plt.scatter, 'total_bill', 'tip')
x = x.add_legend()
x = sns.FacetGrid(tips, col='day')
x = x.map(sns.boxplot, 'total_bill', 'time')
x = sns.FacetGrid(tips, col='day', size=4, aspect=1)
x = x.map(sns.boxplot, 'time', 'total_bill')
x = sns.FacetGrid(tips, col='day', col_order=['Thur', 'Fri', 'Sat', 'Sun'], 
                  size=4, aspect=0.4)
x = x.map(sns.boxplot, 'time', 'total_bill', color='red')
x = sns.FacetGrid(tips, col='time', hue='smoker', palette='husl')
x = x.map(plt.scatter, 'total_bill', 'tip')
x = x.add_legend()
tips = sns.load_dataset('tips')
tips.head()
iris = sns.load_dataset('iris')
iris.head()
sns.jointplot(x='total_bill', y='tip', data=tips)
sns.jointplot(x='sepal_length', y='sepal_width', data=iris)
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg',
             color='green')
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')
sns.jointplot(x='sepal_length', y='sepal_width', data=iris, kind='kde')
from scipy.stats import spearmanr
sns.jointplot(x='total_bill', y='size', data=tips)
sns.jointplot(x='total_bill', y='size', data=tips, stat_func=spearmanr)
sns.jointplot(x='total_bill', y='size', data=tips, ratio=4, size=5)
iris = sns.load_dataset('iris')
sns.pairplot(iris)
tips = sns.load_dataset('tips')
sns.pairplot(tips)
sns.pairplot(iris, hue='species')
sns.pairplot(iris, hue='species', palette='husl', markers=['o', 'D', 's'])
sns.pairplot(iris, vars=['sepal_length', 'sepal_width'])
sns.pairplot(iris, size=3, vars=['sepal_length', 'sepal_width'])
sns.pairplot(iris, x_vars=['petal_length', 'petal_width'], 
             y_vars=['sepal_length', 'sepal_width'], hue='species')
sns.pairplot(iris, diag_kind='kde', palette='husl', hue='species')
sns.pairplot(iris, diag_kind='kde', palette='husl', hue='species',
            kind='reg')