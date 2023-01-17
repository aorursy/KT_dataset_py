# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df = pd.read_csv('../input/Mall_Customers.csv')

df.head()
df.isnull().sum()
df.dtypes
fig, axes = plt.subplots(1, 2, figsize=(12,8))



sns.boxplot(x="Gender", y="Annual Income (k$)", data=df, orient='v' , ax=axes[0])

sns.boxplot(x="Gender", y="Spending Score (1-100)", data=df, orient='v' , ax=axes[1])
df_group_one = df[['Gender','Annual Income (k$)','Spending Score (1-100)']]

df_group_one.groupby(['Gender'],as_index=False).mean()
df_female = df[df['Gender'] == "Female"]

print(df_female.shape)

df_female.head()
Percentage = (df_female.shape[0]/df.shape[0])*100

print('Female Percentage: ', round(Percentage), '%')
from sklearn.cluster import DBSCAN

import sklearn.utils

from sklearn.preprocessing import StandardScaler



Clus_dataSet = df_female[['Age','Annual Income (k$)','Spending Score (1-100)']]

Clus_dataSet = np.nan_to_num(Clus_dataSet)

Clus_dataSet = np.array(Clus_dataSet, dtype=np.float64)

Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)



# Compute DBSCAN

db = DBSCAN(eps=0.5, min_samples=4).fit(Clus_dataSet)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_

df_female['Clus_Db']=labels



realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)

clusterNum = len(set(labels)) 



# A sample of clusters

print(df_female[['Age','Annual Income (k$)','Spending Score (1-100)','Clus_Db']].head())



# number of labels

print("number of labels: ", set(labels))
# Black removed and is used for noise instead.

unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = Clus_dataSet[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = Clus_dataSet[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title('Estimated number of clusters: %d' % realClusterNum)

plt.show()



n_noise_ = list(labels).count(-1)

print('number of noise(s): ', n_noise_)
#Visualization

for clust_number in set(labels):

    clust_set = df_female[df_female.Clus_Db == clust_number]

    if clust_number != -1:

        print ("Cluster "+str(clust_number)+', Avg Age: '+ str(round(np.mean(clust_set.Age)))+\

               ', Avg Income: '+ str(round(np.mean(clust_set['Annual Income (k$)'])))+\

               ', Avg Spending: '+ str(round(np.mean(clust_set['Spending Score (1-100)'])))+', Count: '+ str(np.count_nonzero(clust_set.index)))