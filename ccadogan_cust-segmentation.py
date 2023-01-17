# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.datasets.samples_generator import make_blobs 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load data

cust_df = pd.read_csv('/kaggle/input/custdatasets/Cust_Segmentation.csv')

cust_df.head()
#remove the address columns

df = cust_df.drop('Address', axis = 1)

df.head()
#Preprocess data - normalize data

X = df.values[:,1:]

X = np.nan_to_num(X)

Clust_dataSet = StandardScaler().fit_transform(X)

Clust_dataSet
# apply k-means on dataset and get cluster labels

clusNum = 3 

k_means = KMeans(init = 'k-means++', n_clusters = clusNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(np.unique(labels))

#add new labels to dataframe

df['Clus_kmeans'] =labels

df.head(5)
X[:5]
#check the centroid values by averaging the features in each cluster

df.groupby('Clus_kmeans').mean()
#visualize the distribution of customers based on their age and sex



area = np.pi*(df.Edu)**2

plt.scatter(df.Age, df.Income, s=area, c= labels.astype(np.float), alpha = 0.5 )

plt.xlabel('Age',fontsize = 18)

plt.ylabel('Income', fontsize = 16)



plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize = (8,6))

plt.clf()

ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)



plt.cla()



ax.set_xlabel('Education')

ax.set_ylabel('Age')

ax.set_zlabel('Income')



ax.scatter(df.Edu, df.Age, df.Income, c=labels.astype(np.float))