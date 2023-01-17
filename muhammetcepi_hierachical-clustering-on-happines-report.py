# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import ndimage 

from scipy.cluster import hierarchy 

from scipy.spatial import distance_matrix 

from matplotlib import pyplot as plt 

from sklearn import manifold, datasets 

from sklearn.cluster import AgglomerativeClustering 

from sklearn.datasets.samples_generator import make_blobs 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/world-happiness/2015.csv")

df.head()
featureset = df[["Standard Error","Economy (GDP per Capita)","Family","Health (Life Expectancy)","Freedom","Trust (Government Corruption)","Generosity","Dystopia Residual"]]
from sklearn.preprocessing import MinMaxScaler

x = featureset.values #returns a numpy array

min_max_scaler = MinMaxScaler()

feature_mtx = min_max_scaler.fit_transform(x)

feature_mtx [0:5]
import scipy

leng = feature_mtx.shape[0]

D = scipy.zeros([leng,leng])

for i in range(leng):

    for j in range(leng):

        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
import pylab

import scipy.cluster.hierarchy

Z = hierarchy.linkage(D, 'complete')
from scipy.cluster.hierarchy import fcluster

max_d = 3

clusters = fcluster(Z, max_d, criterion='distance')

clusters
from scipy.cluster.hierarchy import fcluster

k = 5

clusters = fcluster(Z, k, criterion='maxclust')

clusters
fig = pylab.figure(figsize=(20,200))

def llf(id):

    return '[%s ,%s,%s ]' % (df['Country'][id], df['Region'][id],int(float(df['Happiness Rank'][id])) )

    

dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
dist_matrix = distance_matrix(feature_mtx,feature_mtx) 

print(dist_matrix)
agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'complete')

agglom.fit(feature_mtx)

agglom.labels_
df['cluster_'] = agglom.labels_

df.head()
import matplotlib.cm as cm

n_clusters = max(agglom.labels_)+1

colors = cm.rainbow(np.linspace(0, 1, n_clusters))

cluster_labels = list(range(0, n_clusters))





plt.figure(figsize=(15,15))



for color, label in zip(colors, cluster_labels):

    subset = df[df.cluster_ == label]

    for i in subset.index:

            plt.text(subset["Happiness Score"][i], subset.Region[i],str(subset.Country[i]), rotation=25) 

    plt.scatter(subset["Happiness Score"], subset.Region,  c=color, label='cluster'+str(label),alpha=0.5)

#    plt.scatter(subset.horsepow, subset.mpg)

plt.legend()

plt.title('Clusters')

plt.xlabel('Happiness Score')

plt.ylabel('Region')
df.groupby(['cluster_','Region'])['cluster_'].count()
agg_reg = df.groupby(['cluster_','Region'])['Happiness Score','Economy (GDP per Capita)','Freedom','Health (Life Expectancy)'].mean()

agg_reg
for label in cluster_labels:

    subset=agg_reg.loc[(label,),]

    print(subset)
plt.figure(figsize=(15,15))

for color, label in zip(colors, cluster_labels):

    subset = agg_reg.loc[(label,),]

    for i in subset.index:

        plt.text(subset.loc[i][0], subset.loc[i][2], 'Region='+str(i) + ', Health='+str(subset.loc[i][3]))

    plt.scatter(subset["Happiness Score"], subset["Freedom"], c=color, label='cluster'+str(label))

plt.legend()

plt.title('Clusters')

plt.xlabel('horsepow')

plt.ylabel('mpg')