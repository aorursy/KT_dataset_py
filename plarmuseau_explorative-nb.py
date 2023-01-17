import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

outlet = pd.read_csv("../input/outlets.csv")

outlet.describe().T
import matplotlib

import matplotlib.pyplot as plt



allLat  = np.array(list(outlet['Latitude']) )

allLong = np.array(list(outlet['Longitude']))



latRange = [49.5,51.475]

longRange = [2.59,6.34]

# show the log density of pickup and dropoff locations

imageSize = (350,350)



allLatInds  = imageSize[0] - (imageSize[0] * (allLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)

allLongInds =                (imageSize[1] * (allLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)





locationDensityImage = np.zeros(imageSize)

for latInd, longInd in zip(allLatInds,allLongInds):

    locationDensityImage[latInd,longInd] += 1



fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))

ax.imshow(np.log(locationDensityImage+1),cmap='hot')

ax.set_axis_off()
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier



loc_df = pd.DataFrame()

loc_df['longitude'] = outlet.Longitude

loc_df['latitude'] = outlet.Latitude

kmeans = KMeans(n_clusters=165, random_state=1, n_init = 10).fit(loc_df)



loc_df['label'] = kmeans.labels_

outlet['cluster_day']=kmeans.labels_

plt.figure(figsize = (18,12))

for label in loc_df.label.unique():

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize =2)

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')

    plt.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')

    plt.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 10)





plt.title('Day Cluster Belgium')

plt.show()
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier



loc_df = pd.DataFrame()

loc_df['longitude'] = outlet.Longitude

loc_df['latitude'] = outlet.Latitude

kmeans = KMeans(n_clusters=82, random_state=1, n_init = 10).fit(loc_df)



loc_df['label'] = kmeans.labels_

outlet['cluster_night']=kmeans.labels_

plt.figure(figsize = (18,12))

for label in loc_df.label.unique():

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize =2)

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')

    plt.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')

    plt.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 10)



plt.title('Night Cluster Belgium')

plt.show()