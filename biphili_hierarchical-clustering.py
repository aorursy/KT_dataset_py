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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

import warnings 

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/k-means-mall-custome-data/Mall_Customers_1.csv')

df.head()
X=df.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(X,method='ward')) # Within cluster variance is reduced with ward method

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(X)
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='magenta',label='Low spenders ')

plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Young Average Spenders')

plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Old High Spenders')

plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Young High Spenders')

plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='burlywood',label='Sensible')

#plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=100,c='blue',label='Sensible')

#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='Centroids')

plt.title('Cluster of Clients')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.ioff()

plt.show()