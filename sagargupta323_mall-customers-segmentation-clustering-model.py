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
from sklearn import linear_model
# For interactive visualizations
import seaborn as sns
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
plt.subplot(1, 2, 1)
sns.distplot(data['Annual Income (k$)'],color='orange')
plt.title('Distribution of Annual Income ', fontsize = 15)
plt.xlabel('Annual Income(in k$)')
plt.ylabel('Prob.')


plt.subplot(1, 2, 2)
sns.distplot(data['Age'])
plt.title('Distribution of Age', fontsize = 15)
plt.xlabel('Range of Age')
plt.ylabel('prob.')

plt.show()
labels = ['Female', 'Male']
size = data['Gender'].value_counts()
colors = ['blue', 'orange']

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, labels = labels,autopct = '%.2f%%')
plt.title('Gender', fontsize = 15)
plt.legend(fontsize=12)
plt.show()
plt.rcParams['figure.figsize'] = (14, 8)
sns.countplot(data['Age'])
plt.title('Distribution of Age', fontsize = 15)
plt.show()
plt.rcParams['figure.figsize'] = (14, 8)
sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')
plt.title('Distribution of Annual Income', fontsize = 15)
plt.show()
plt.subplot(1,2,1)
sns.countplot(data['Spending Score (1-100)'])
plt.title('Distribution Grpah of Spending Score', fontsize = 15)

plt.subplot(1,2,2)
sns.distplot(data['Spending Score (1-100)'])
sns.pairplot(data)
plt.title('Pairplot for the Data', fontsize = 15)
plt.show()
data.corr()
## Correlation coeffecients heatmap
sns.heatmap(data.corr(), annot=True).set_title('Correlation Factors Heat Map', size='15')
plt.rcParams['figure.figsize'] = (16, 8)
sns.stripplot(data['Gender'], data['Age'], size = 10)
plt.title('Gender vs Spending Score', fontsize = 15)
plt.show()
plt.rcParams['figure.figsize'] = (16,8)
sns.violinplot(data['Gender'], data['Annual Income (k$)'])
plt.title('Gender vs Spending Score', fontsize = 15)
plt.show()
data.head()
x = data[['Annual Income (k$)','Spending Score (1-100)']]

# let's check the shape of x
print(x.shape)
x
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km= KMeans(i)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()
km2=KMeans(5)
km2.fit(x)
data_clustered=data.copy()
data_clustered['Cluster']=km2.fit_predict(x)
data_clustered.head()
plt.rcParams['figure.figsize'] = (10,5)
plt.scatter(data_clustered['Annual Income (k$)'],data_clustered['Spending Score (1-100)'],c=data_clustered['Cluster'],cmap='rainbow')
plt.title('K Means Clustering', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
x = data.iloc[:, [2, 4]].values
x.shape
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.rcParams['figure.figsize'] = (16, 8)
plt.plot(wcss)
plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid()
plt.show()
kmeans = KMeans(4)
ymeans = kmeans.fit_predict(x)

plt.rcParams['figure.figsize'] = (16, 8)
plt.title('Cluster of Ages', fontsize = 30)

plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'green', label = 'Usual Customers' )
plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')
plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'cyan', label = 'Target Customers(Young)')
plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'blue')

plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
