# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)
# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
mall= pd.read_csv(r"/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

mall.head()
mall.shape
mall.info()
mall.describe()
mall_d= mall.copy()

mall_d.drop_duplicates(subset=None,inplace=True)
mall_d.shape
mall.shape
(mall.isnull().sum() * 100 / len(mall)).value_counts(ascending=False)
mall.isnull().sum()
(mall.isnull().sum(axis=1) * 100 / len(mall)).value_counts(ascending=False)
mall.isnull().sum(axis=1).value_counts(ascending=False)
plt.figure(figsize = (5,5))

gender = mall['Gender'].sort_values(ascending = False)

ax = sns.countplot(x='Gender', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

plt.show()
 

plt.figure(figsize = (20,5))

gender = mall['Age'].sort_values(ascending = False)

ax = sns.countplot(x='Age', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
plt.figure(figsize = (25,5))

gender = mall['Annual Income (k$)'].sort_values(ascending = False)

ax = sns.countplot(x='Annual Income (k$)', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
plt.figure(figsize = (27,5))

gender = mall['Spending Score (1-100)'].sort_values(ascending = False)

ax = sns.countplot(x='Spending Score (1-100)', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (5,5))

sns.heatmap(mall.corr(), annot = True, cmap="rainbow")

plt.savefig('Correlation')

plt.show()
sns.pairplot(mall,corner=True,diag_kind="kde")

plt.show()
# Data before Outlier Treatment 

mall.describe()
f, axes = plt.subplots(1,3, figsize=(15,5))

s=sns.violinplot(y=mall.Age,ax=axes[0])

axes[0].set_title('Age')

s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])

axes[1].set_title('Annual Income (k$)')

s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])

axes[2].set_title('Spending Score (1-100)')

plt.show()

Q3 = mall['Annual Income (k$)'].quantile(0.99)

Q1 = mall['Annual Income (k$)'].quantile(0.01)

mall['Annual Income (k$)'][mall['Annual Income (k$)']<=Q1]=Q1

mall['Annual Income (k$)'][mall['Annual Income (k$)']>=Q3]=Q3
# Data After Outlier Treatment 

mall.describe()
f, axes = plt.subplots(1,3, figsize=(15,5))

s=sns.violinplot(y=mall.Age,ax=axes[0])

axes[0].set_title('Age')

s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])

axes[1].set_title('Annual Income (k$)')

s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])

axes[2].set_title('Spending Score (1-100)')

plt.show()
# Dropping CustomerID,Gender field to form cluster



mall_c = mall.drop(['CustomerID','Gender'],axis=1,inplace=True)
mall.head()