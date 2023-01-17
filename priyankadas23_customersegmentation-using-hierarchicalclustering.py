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
import pandas as pd

df=pd.read_csv('../input/mall-customers/Mall_Customers.csv')

df.head()
df.dropna(axis=0)
df=df.drop(['CustomerID','Genre'],axis=1)
import numpy as np

data=np.asarray(df)

data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

data=sc.fit_transform(data)

data
from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

ac.fit(data)

labels=ac.labels_
import scipy.cluster.hierarchy as hc

dendrogram = hc.dendrogram(hc.linkage(data, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()

# Visualising the clusters

import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(data[labels==0, 0], data[labels==0, 1], s=50, marker='o', color='red')

plt.scatter(data[labels==1, 0], data[labels==1, 1], s=50, marker='o', color='blue')

plt.scatter(data[labels==2, 0], data[labels==2, 1], s=50, marker='o', color='green')

plt.scatter(data[labels==3, 0], data[labels==3, 1], s=50, marker='o', color='purple')

plt.scatter(data[labels==4, 0], data[labels==4, 1], s=50, marker='o', color='orange')

plt.show()