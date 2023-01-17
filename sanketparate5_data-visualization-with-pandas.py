import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from numpy.random import randn, randint, uniform, sample
df = pd.DataFrame(randn(1000), index = pd.date_range('2019-06-07', periods = 1000), columns=['value'])
ts = pd.Series(randn(1000), index = pd.date_range('2019-06-07', periods = 1000))
df.head()
df['value'] = df['value'].cumsum()
df.head()
ts = ts.cumsum()
ts.head()
type(df), type(ts)
ts.plot(figsize=(5,5))
plt.plot(ts)
df.plot()
iris = sns.load_dataset('iris')
iris.head()
ax = iris.plot(figsize=(15,8), title='Iris Dataset')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ts.plot(kind = 'bar')
plt.show()
df = iris.drop(['species'], axis = 1)
df.iloc[0]
df.iloc[0].plot(kind='bar')
df.iloc[0].plot.bar()
titanic = sns.load_dataset('titanic')
titanic.head()
titanic['pclass'].plot(kind = 'bar')
df = pd.DataFrame(randn(10, 4), columns=['a', 'b', 'c', 'd'])
df.head(10)
df.plot.bar()
df.plot(kind = 'bar')
df.plot.barh()
iris.plot.hist()
iris.plot(kind = 'hist')
iris.plot(kind = 'hist', stacked = False, bins = 100)
iris.plot(kind = 'hist', stacked = True, bins = 50, orientation = 'horizontal')
iris['sepal_width'].diff()
iris['sepal_width'].diff().plot(kind = 'hist', stacked = True, bins = 50)

df = iris.drop(['species'], axis = 1)
df.diff().head()
df.diff().hist(color = 'b', alpha = 0.1, figsize=(10,10))
color = {'boxes': 'DarkGreen', 'whiskers': 'b'}
color
df.plot.scatter(x = 'sepal_length', y = 'petal_length')
df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width')
df.head()
df.plot.scatter(x = 'sepal_length', y = 'petal_length', label = 'Length');
#df.plot.scatter(x = 'sepal_width', y = 'petal_width', label = 'Width', ax = ax, color = 'r')
#df.plot.scatter(x = 'sepal_width', y = 'petal_length', label = 'Width', ax = ax, color = 'g')
df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width', s = 190)
df.plot.hexbin(x = 'sepal_length', y = 'petal_length', gridsize = 5, C = 'sepal_width')
d = df.iloc[0]
d
d.plot.pie(figsize = (10,10))
d = df.head(3).T
d.plot.pie(subplots = True, figsize = (20, 20))
d.plot.pie(subplots = True, figsize = (35, 25), fontsize = 26, autopct = '%.2f')
plt.show()
[0.1]*4
series = pd.Series([0.2]*5, index = ['a','b','c', 'd','e'], name = 'Pie Plot')
series.plot.pie()
plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize= (8,8), diagonal='kde', color = 'b')
plt.show()
ts.plot.kde()
from pandas.plotting import andrews_curves
andrews_curves(df, 'sepal_width')
ts.plot(style = 'r--', label = 'Series', legend = True)
plt.show()
df.plot(legend = True, figsize = (10, 5), logy = True)
plt.show()
x = df.drop(['sepal_width', 'petal_width'], axis = 1)
x.head()
y = df.drop(['sepal_length', 'petal_length'], axis = 1)
y.head()
ax = x.plot()
y.plot(figsize = (16,10), secondary_y=True, ax = ax)
plt.show()
x.plot(figsize=(10,5), x_compat = True)
plt.show()
df.plot(subplots = True)
plt.show()

df.plot(subplots = True, sharex = False, layout = (2,3), figsize = (16,8))
plt.tight_layout()
plt.show()