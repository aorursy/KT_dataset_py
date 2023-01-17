# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for path

import os

print(os.listdir("../input"))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
# For interactive visualizations

import seaborn as sns

plt.style.use('fivethirtyeight')

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff
# importing the dataset

data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

data.shape
# Let's see top 5 dataset

data.head()
# Let's see last 5 datasets

data.tail()
data.info()
data.describe()
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (14, 6)



plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(data['Annual Income (k$)'])

plt.title('Distribution of Annual Income', fontsize = 15)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')





plt.subplot(1, 2, 2)

sns.set(style = 'whitegrid')

sns.distplot(data['Age'], color = 'orange')

plt.title('Distribution of Age', fontsize = 15)

plt.xlabel('Range of Age')

plt.ylabel('Count')

plt.show()
labels = ['Female', 'Male']

size = data['Gender'].value_counts()

colors = ['blue', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 15)

plt.axis('off')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (14, 8)

sns.countplot(data['Age'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (14, 8)

sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Distribution of Annual Income', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (14, 8)

sns.countplot(data['Spending Score (1-100)'], palette = 'copper')

plt.title('Distribution Grpah of Spending Score', fontsize = 15)

plt.show()
sns.pairplot(data)

plt.title('Pairplot for the Data', fontsize = 15)

plt.show()
## Correlation coeffecients heatmap

sns.heatmap(data.corr(), annot=True).set_title('Correlation Factors Heat Map', size='15')
plt.rcParams['figure.figsize'] = (16, 8)

sns.stripplot(data['Gender'], data['Age'], palette = 'Purples', size = 10)

plt.title('Gender vs Spending Score', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (16,8)

sns.violinplot(data['Gender'], data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Gender vs Spending Score', fontsize = 15)

plt.show()
x = data['Annual Income (k$)']

y = data['Age']

z = data['Spending Score (1-100)']



sns.lineplot(x, y, color = 'green')

sns.lineplot(x, z, color = 'orange')

plt.title('Annual Income vs Age and Spending Score', fontsize = 15)

plt.show()
x = data.iloc[:, [3, 4]].values



# let's check the shape of x

print(x.shape)
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'green', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 15)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()
import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendrogam', fontsize = 15)

plt.xlabel('Customers')

plt.ylabel('Ecuclidean Distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)



plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'green', label = 'miser')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 15)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.grid()

plt.show()
x = data.iloc[:, [2, 4]].values

x.shape
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (16, 8)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.grid()

plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)



plt.rcParams['figure.figsize'] = (16, 8)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'green', label = 'Usual Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'cyan', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'blue')



plt.style.use('fivethirtyeight')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.grid()

plt.show()