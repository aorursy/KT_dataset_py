# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch 
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/airline-data/EastWestAirlines.csv")
data.head()
np.any(np.isnan(data))
np.all(np.isfinite(data))
np.where(np.isnan(data))
data.loc[(data['Bonus_miles']<=5000),'Group']=1
data.loc[(data['Bonus_miles']>5000) & (data['Bonus_miles']<=10000),'Group']=2
data.loc[(data['Bonus_miles']>10000) & (data['Bonus_miles']<=25000),'Group']=3
data.loc[(data['Bonus_miles']>25000) & (data['Bonus_miles']<=50000),'Group']=4
data.loc[(data['Bonus_miles']>50000),'Group']=5
data['Group'].astype('int')
data.head()
data[np.any(np.isnan(data),axis=1)]
data_h=data # This for hierarchical Clustering 
data_k=data # This for K-Means Clustering
kmeans=KMeans(n_clusters=5)
kmeans.fit(data_k)
kmeans.cluster_centers_
pred=kmeans.predict(data_k)
pred
data_k['Cluster']=pd.DataFrame(pred)
data_k.head()
data_k[data_k['Cluster']==0] # Cluster=0
data_k[data_k['Cluster']==1]  # Cluster=1
data_k[data_k['Cluster']==2]  # Cluster=2
data_k[data_k['Cluster']==3]  # Cluster=3
data_k[data_k['Cluster']==4]  # Cluster=4
scatter=plt.scatter(data_k['Group'],data_k['Cluster'],c=data_k['Group'])
plt.legend(*scatter.legend_elements(),bbox_to_anchor=(1.07, 1))
plt.show()
for x in data_k['Cluster'].unique():
    scatter=plt.scatter(data_k['Bonus_miles'],data_k['Cluster']==x,c=data_k['Group']).legend_elements()
    plt.legend(*scatter,bbox_to_anchor=(1.21, 1))
    plt.show()
for x in data_k['Cluster'].unique():
    scatter=plt.scatter(data_k['Award?'],data_k['Cluster']==x,c=data_k['Group']).legend_elements()
    plt.legend(*scatter,bbox_to_anchor=(1.21, 1))
    plt.show()
for x in data_k['Cluster'].unique():
    scatter=plt.scatter(data_k['cc1_miles'],data_k['Cluster']==x,c=data_k['Group']).legend_elements()
    plt.legend(*scatter,bbox_to_anchor=(1.21, 1))
    plt.show()
for x in data_k['Cluster'].unique():
    scatter=plt.scatter(data_k['cc2_miles'],data_k['Cluster']==x,c=data_k['Group']).legend_elements()
    plt.legend(*scatter,bbox_to_anchor=(1.21, 1))
    plt.show()
for x in data_k['Cluster'].unique():
    scatter=plt.scatter(data_k['cc3_miles'],data_k['Cluster']==x,c=data_k['Group']).legend_elements()
    plt.legend(*scatter,bbox_to_anchor=(1.21, 1))
    plt.show()
hclust=sch.dendrogram(sch.linkage(data_h,method='ward'))