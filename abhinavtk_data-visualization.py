# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Using iris dataset
iris = sns.load_dataset('iris')
iris.head()
iris.plot(figsize=(10,5), title='Iris dataset')

iris = iris.drop('species', axis=1)
iris.iloc[0].plot.bar()
# Horizontal bar plot
iris.plot.barh()
iris.plot.hist()
iris.plot(kind = 'hist', bins = 100, stacked=True)
iris.plot.scatter(x='sepal_length', y='sepal_width', c='petal_length')
iris.plot.scatter(x='sepal_length', y='petal_length',c='sepal_width', label='Length', s=50)
#Pie chart
df = iris.iloc[0]
df.plot.pie()
d = iris.head(3).T
d
d.plot.pie(subplots=True, figsize=(30,30))
d.plot.pie(subplots = True, figsize = (30, 30), fontsize = 20, autopct = '%.2f')
from pandas.plotting import scatter_matrix
scatter_matrix(iris, figsize=(10,10), color='r')
scatter_matrix(iris, figsize=(10,10), diagonal='kde', color='g')
# Andrew's curves
from pandas.plotting import andrews_curves
andrews_curves(iris,'sepal_width')
iris.plot(legend = True, figsize = (10, 5), logy = True)
iris
x = iris.drop(['sepal_width', 'petal_width'], axis=1)
y = iris.drop(['sepal_length', 'petal_length'], axis=1)
ax = x.plot()
ax
y.plot(figsize=(16,10), secondary_y = True)
iris.plot(subplots=True)
iris.plot(subplots=True,sharex = False, layout=(2,3), figsize=(16,8))
plt.tight_layout()
