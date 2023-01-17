import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy  # draw dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt 

import sklearn
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df_airbnb.sample(10)
len(df_airbnb)
df_airbnb.isnull().sum()
#replacing all NaN values in 'reviews_per_month' with 0
df_airbnb.fillna({'reviews_per_month':0}, inplace=True)
#replacing all NaN values in 'last_review' with 0
df_airbnb.fillna({'last_review':0}, inplace=True)
df_airbnb.isnull().sum()
len(df_airbnb.neighbourhood_group.unique())
df_airbnb.neighbourhood_group.unique()
#len(df_airbnb.neighbourhood.unique())
df_airbnb.neighbourhood.unique()
len(df_airbnb.room_type.unique())
df_airbnb.room_type.unique()
df_airbnb.info()
#replace neighbourhood_group number
df_airbnb['neighbourhood_group'].replace({
'Brooklyn':1, 'Manhattan':2, 'Queens':3, 'Staten Island':4, 'Bronx':5
}, inplace=True)
df_airbnb.neighbourhood_group.unique()
#replace room_type number
df_airbnb['room_type'].replace({'Private room':1, 'Entire home/apt':2, 'Shared room':3}, inplace=True)
#drop unuse columns
X = df_airbnb.drop(columns=['id','name','host_name','neighbourhood','latitude','longitude','last_review'])
# sample row
X = X.sample(5000)
X
#select column Y
Y = df_airbnb['availability_365']
# sample row
Y = Y.sample(5000)
Y
Z = linkage(X,'complete') # ward, complete, average
dendrogram(Z, p=12,leaf_rotation=45.,show_contracted=True)
plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axvline(x=250)

plt.show()
k = 2

Hclustering = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='ward')
Hclustering.fit(X)
sm.accuracy_score(Y,Hclustering.labels_)
Hclustering = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='complete')
Hclustering.fit(X)
sm.accuracy_score(Y,Hclustering.labels_)
Hclustering = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='average')
Hclustering.fit(X)
sm.accuracy_score(Y,Hclustering.labels_)