import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

rawdata = pd.read_csv('../input/sample_table.csv')

rawdata=rawdata[rawdata['total_cars']==0]

#rawdata=rawdata[:100]

rawdata.head()
rawdata=rawdata.reset_index(drop=True)

print("Amount of running out of car report : ",len(rawdata))

print("Amount of unique time : ",len(rawdata['timestamp'].unique()))
rawdata.head()
locationset=set()

for i in range(len(rawdata)):

    locationset.add((rawdata.latitude[i],rawdata.longitude[i]))

print("Amount of places : ",len(locationset))
locationset=list(locationset)
countfolocation=np.zeros(len(locationset))

for i in range(len(rawdata)):

    countfolocation[locationset.index((rawdata.latitude[i],rawdata.longitude[i]))]+=1

print("First five of frequency count of each location")

countfolocation=list(countfolocation)

print(countfolocation[:5])
for i in range(len(countfolocation)-1,-1,-1):

    if countfolocation[i]<12000:

        del countfolocation[i]

        del locationset[i]
x=list((locationset[i][0]) for i in range(len(locationset)))

y=list((locationset[i][1]) for i in range(len(locationset)))

import folium      #  folium libraries

from   folium.plugins import MarkerCluster

from statistics import mean

map_world = folium.Map(location=[mean(x), mean(y)], tiles = 'OpenStreetMap', zoom_start = 12)



#  add Locations to map

for i in range(len(x)):

    folium.CircleMarker(

        [x[i], y[i]],

        radius=4*(countfolocation[i]/10000),

        popup=countfolocation[i],

        fill=True,

        color='Red',

        fill_color='Red',

        fill_opacity=0.6

        ).add_to(map_world)



#  display interactive map

map_world
from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

X = np.array(list(zip(x, y))).reshape(len(x), 2)

distortions = []

K = range(1,15)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(X)

    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

#distortions
plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
kmeanModel = KMeans(n_clusters=6).fit(X)

centers = kmeanModel.cluster_centers_

y_kmeans = kmeanModel.predict(X)

print("The nearest new praking lot location for first five location : ",y_kmeans[:5])
#print(len(y_kmeans),len(x),len(X))

print("Location of each new parking lot in order index")

print(centers)
newstation=np.zeros(6)

for i in range(len(y_kmeans)):

    newstation[y_kmeans[i]]+=countfolocation[i]

print("Counting for the number of the report rely on each new location")

print(newstation)
import numpy

sortedstation=(numpy.argsort(newstation))

print("Sorted index by frequency of report near by each location")

print(sortedstation)
sortedcenter=[]

print("Sorted centers weight by high amough of nearby report")

for i in range(len(sortedstation)-1,-1,-1):

    sortedcenter.append(list(centers[sortedstation[i]]))

    print(sortedcenter[len(sortedcenter)-1])

#print(sortedcenter)
map_world = folium.Map(location=[mean(x), mean(y)], tiles = 'OpenStreetMap', zoom_start = 12)



for i in range(len(centers)):

    folium.CircleMarker(

        [sortedcenter[i][0], sortedcenter[i][1]],

        radius=4*(9-i),

        popup=i+1,

        fill=True,

        color='Green',

        fill_color='Green',

        fill_opacity=0.2

        ).add_to(map_world)

    

for i in range(len(centers)):

    folium.CircleMarker(

        [sortedcenter[i][0], sortedcenter[i][1]],

        radius=2,

        popup=i+1,

        fill=True,

        color='Blue',

        fill_color='Blue',

        fill_opacity=0.6

        ).add_to(map_world)

    

map_world