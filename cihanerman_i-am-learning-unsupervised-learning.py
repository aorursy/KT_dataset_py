# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import data
data = pd.read_csv('../input/column_2C_weka.csv')
# data info
data.info()
# data descript
data.describe()
# data ploting
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
# kmeans clustring
# The best k value find
data2 = data.drop('class',axis=1)
k_values_list = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    k_values_list.append(kmeans.inertia_)

plt.plot(range(1,10),k_values_list)
plt.xlabel('number of k (cluster) value')
plt.ylabel('wcss')
plt.show()
# model for  k = 2
kmeans = KMeans(n_clusters = 2)
labels = kmeans.fit_predict(data2)
plt.scatter(data2['pelvic_radius'],data2['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
# standardization
data3 = data.drop('class',axis = 1)
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
labels = pipe.fit_predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
# dendrogram
merge = linkage(data3.iloc[200:220,:],method = 'ward')
dendrogram(merge, leaf_rotation = 90, leaf_font_size = 6)
plt.xlabel('dendogram')
plt.ylabel('eucliden distance')
plt.show()
# hierarchy cluster model
hierarchy_cluster = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
clusters = hierarchy_cluster.fit_predict(data3)
#print (clusters)
data3['label'] = clusters
# hierarchy cluster ploting data3 pelvic_radius and degree_spondylolisthesis
plt.scatter(data3['pelvic_radius'][data3.label == 0],data3['degree_spondylolisthesis'][data3.label == 0], color='yellow')
plt.scatter(data3['pelvic_radius'][data3.label == 1],data3['degree_spondylolisthesis'][data3.label == 1], color='blue')
plt.show()