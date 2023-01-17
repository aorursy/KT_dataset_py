# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #seaborn

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
iris.head()
iris.info()
iris['species'].value_counts()
iris.describe()
sns.FacetGrid(iris,hue='species',size=7).map(sns.distplot,'sepal_length').add_legend()
sns.FacetGrid(iris,hue='species',height=7).map(sns.distplot,'sepal_width').add_legend()
sns.FacetGrid(iris,hue='species',size=7).map(sns.distplot,'petal_length').add_legend()
sns.FacetGrid(iris,hue='species',size=7).map(sns.distplot,'petal_width').add_legend()
count,bin_edges = np.histogram(iris[iris['species']=='Iris-versicolor']['petal_length'])

plt.plot(bin_edges[1:],count/(sum(count)))

plt.plot(bin_edges[1:],np.cumsum(count/(sum(count))))



count,bin_edges = np.histogram(iris[iris['species']=='Iris-virginica']['petal_length'])

plt.plot(bin_edges[1:],count/(sum(count)))

plt.plot(bin_edges[1:],np.cumsum(count/(sum(count))))

plt.grid()
count,bin_edges = np.histogram(iris[iris['species']=='Iris-versicolor']['petal_width'])

plt.plot(bin_edges[1:],count/(sum(count)))

plt.plot(bin_edges[1:],np.cumsum(count/(sum(count))))



count,bin_edges = np.histogram(iris[iris['species']=='Iris-virginica']['petal_width'])

plt.plot(bin_edges[1:],count/(sum(count)))

plt.plot(bin_edges[1:],np.cumsum(count/(sum(count))))

plt.grid()
iris_setosa = iris.loc[iris['species']=='Iris-setosa']
print(np.mean(iris_setosa['petal_length']))
print(np.mean(np.append(iris_setosa['petal_length'],50)))
print(np.std(iris_setosa['petal_length']))
print(np.median(iris_setosa['petal_length']))
print(np.median(np.append(iris_setosa['petal_length'],50)))
sns.boxplot(x='species',y='sepal_length',data=iris)
sns.boxplot(x='species',y='sepal_width',data=iris)
sns.boxplot(x='species',y='petal_length',data=iris)
sns.boxplot(x='species',y='petal_width',data=iris)
sns.violinplot(x='species',y='petal_length',size=7,data=iris)
sns.lmplot('petal_length','petal_width',hue='species',data=iris,fit_reg=False)
sns.pairplot(iris,hue='species')