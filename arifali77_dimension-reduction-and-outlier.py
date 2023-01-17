# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.decomposition import FactorAnalysis

from sklearn import datasets
iris=datasets.load_iris()

x=iris.data

variable_names= iris.feature_names

x[0:10,]
factor=FactorAnalysis().fit(x)

pd.DataFrame(factor.components_, columns=variable_names)
import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn import decomposition

from sklearn.decomposition import PCA

from pylab import rcParams

from IPython.display import Image

from IPython.core.display import HTML

%matplotlib inline
sns.set_style('whitegrid')

rcParams['figure.figsize']=10,8
pca = decomposition.PCA()

iris_pca = pca.fit_transform(x)

pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()
comps=pd.DataFrame(pca.components_, columns=variable_names)

comps
sns.heatmap(comps)
df=pd.read_csv("../input/iris.csv")

df1=df.drop('Unnamed: 0', axis=1)
df1.boxplot(return_type='dict')

plt.plot()
sepal_width= x[:,1]

iris_outliers = (sepal_width > 4)

df1[iris_outliers]
sepal_width= x[:,1]

iris_outliers = (sepal_width < 2.05)

df1[iris_outliers]
sns.boxplot(x='Species', y='Sepal.Length', data=df1, palette='hls')
sns.pairplot(df1, hue='Species', palette='hls')
from sklearn.cluster import DBSCAN

from collections import Counter
df2=df1.drop('Species', axis=1)
df2.head()
model=DBSCAN(eps=0.8, min_samples=19).fit(df2)

print(model)
outliers_df=pd.DataFrame(df2)

print(Counter(model.labels_))

print( outliers_df[model.labels_ ==  -1])
fig, ax=plt.subplots(1,1)

color=model.labels_

ax.scatter(df2['Petal.Length'], df2['Sepal.Width'], c=color, s= 120)

ax.set_xlabel('Petal length')

ax.set_ylabel('Sepal Width')

plt.title('DBSCAN for Outliers')

plt.show()