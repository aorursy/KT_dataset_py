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
%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap





import warnings 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler, normalize



from sklearn.cluster import KMeans



import scipy.cluster.hierarchy as sch

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import AgglomerativeClustering





from sklearn.decomposition import PCA



from sklearn.metrics import silhouette_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
df = pd.read_csv('../input/ml-training-vlib/Mall_Customers.csv')
df.head()
cols = df.columns

df.columns = [col.lower() for col in cols]



df.rename(columns={'annual income (k$)' : 'ann_income', 

                    'spending score (1-100)' : 'sp_score'},

                   inplace=True)



df.head()
sns.countplot('gender',data=df)
sns.boxplot('gender','age',data=df)
X = df.iloc[:, [3, 4]].values
fig = plt.figure(figsize = (10,5))

plt.scatter(X[:,0],X[:,1],s=100,c='magenta',label='All customers')

plt.title('Clients before clustering')

plt.xlabel('Annual income $')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
#Elbow method

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)
fig = plt.figure(figsize = (15,8))

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='magenta',label='Standard')

plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Careful')

plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Sensible ')

plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='black',label='Careless')

plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='burlywood',label='Target')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='red',label='Centroids')

plt.title('Cluster of Clients')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
dendogram=sch.dendrogram(sch.linkage(X,method='ward')) # Within cluster variance is reduced with ward method

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(X)
fig = plt.figure(figsize = (15,8))

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='magenta',label='Careful')

plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Standard')

plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Target')

plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Careless')

plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='burlywood',label='Sensible')

plt.title('Cluster of Clients')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.ioff()

plt.show()