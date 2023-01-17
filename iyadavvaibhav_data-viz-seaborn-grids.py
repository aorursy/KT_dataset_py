import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
iris = pd.read_csv('../input/iris.csv')
iris.head()
sns.PairGrid(iris)
g = sns.PairGrid(iris)
g.map(plt.scatter)
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_lower(plt.scatter)
g.map_upper(sns.kdeplot)
sns.pairplot(iris,hue='species',palette='rainbow')
# FacetGrid
tips = pd.read_csv('../input/tips.csv')
tips.head()
g = sns.FacetGrid(tips, col="time", row="smoker")
g = sns.FacetGrid(tips, col="time", row="smoker")
g = g.map(plt.hist, "total_bill")
g = sns.JointGrid(x='total_bill', y='tip', data=tips)
g = sns.JointGrid(x='total_bill', y='tip', data=tips)
g = g.plot(sns.regplot, sns.distplot)
