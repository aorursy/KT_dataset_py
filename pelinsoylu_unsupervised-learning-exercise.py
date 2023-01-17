# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings("ignore")
song_data=pd.read_csv("../input/19000-spotify-songs/song_data.csv")
song_data.head()
song_data.info()
#drop the features we don't use.

song_data.drop(["song_name"],axis=1,inplace=True)
song_data.song_duration_ms= song_data.song_duration_ms.astype(float)

song_data.time_signature= song_data.time_signature.astype(float)

song_data.audio_mode= song_data.audio_mode.astype(float)
song_data["popularity"]= [ 1 if i>=70 else 0 for i in song_data.song_popularity ]

song_data["popularity"].value_counts()
data2=song_data.head(1000)

plt.scatter(data2["danceability"],data2["energy"],color="orange")

plt.xlabel("danceability")

plt.ylabel("energy")

plt.show()
#Kmeans Clustering

data3=data2.loc[:,["danceability","energy"]]



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data3)

labels=kmeans.predict(data3) # labels=kmeans.fit_predict(data3)



plt.scatter(data2["danceability"],data2["energy"],c=labels)

plt.xlabel("danceability")

plt.ylabel("energy")

plt.show()

# Cross Tabulation Table

df = pd.DataFrame({'labels':labels,"popularity":data2['popularity']})

ct = pd.crosstab(df['labels'],df['popularity'])

print(ct)
#Inertia

inertia_list = np.empty(8)

for i in range(1,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.ylabel('Inertia')

plt.show()
data4= song_data.drop('popularity',axis = 1)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(data4)

labels = pipe.predict(data4)

df = pd.DataFrame({'labels':labels,"popularity":song_data['popularity']})

ct = pd.crosstab(df['labels'],df['popularity'])

print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram

plt.figure(figsize=[10,10])



merg = linkage(data2.iloc[200:220,:],method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 10)

plt.xlabel("Data Points")

plt.ylabel("Euclidean Distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering

data_new=data2.iloc[:,[2,3]]

data_new["label"]=0

data=data_new



hc_cluster=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")

cluster=hc_cluster.fit_predict(data2)

data["label"]=cluster

plt.scatter(data.acousticness[data.label == 0 ],data.danceability[data.label == 0],color = "red")

plt.scatter(data.acousticness[data.label == 1 ],data.danceability[data.label == 1],color = "green")

plt.scatter(data.acousticness[data.label == 2 ],data.danceability[data.label == 2],color = "blue")

plt.show()

from sklearn.manifold import TSNE

color_list = ['red' if i==1 else 'purple' for i in data2.loc[:,'popularity']]



model = TSNE(learning_rate=100)

transformed = model.fit_transform(data3) #danceability and energy

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list )

plt.xlabel('danceability')

plt.xlabel('energy')

plt.show()
# PCA

from sklearn.decomposition import PCA

model = PCA()

model.fit(data4) 

transformed = model.transform(data4)

print('Principle components: ',model.components_)
# PCA variance

scaler = StandardScaler() 

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(data4)



plt.bar(range(pca.n_components_), pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
#popular vs unpopular songs

color_list = ['yellow' if i==1 else 'teal' for i in data2.loc[:,'popularity']]

pca = PCA(n_components = 2)

pca.fit(data2)

transformed = pca.transform(data2)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list)

plt.show()