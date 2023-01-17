# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/bournemouth-venues"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
venues = pd.read_csv('../input/bournemouth-venues/bournemouth_venues.csv')

venues.head()
# view the total objects and variables in the data set

venues.shape
# view the type of data in the data 

venues.info()
# View the last 10 rows

venues.tail(10)
venues.isnull().sum()
venues.describe().transpose()
# Column count on Discrete data

# Calculate the Column count on Venue Name

venues['Venue Name'].value_counts()
# Calculate the Column count on Venue Category

venues['Venue Category'].value_counts()
# Import Library

from scipy.stats import skew , kurtosis
# Skewness and Kurtosis for -Venue Latitude

print("skewness of the Venue Latitude" , skew(venues['Venue Latitude']))

print("kurtosis of the Venue Latitude" , kurtosis(venues['Venue Latitude']))
# Skewness and Kurtosis for -Venue Longitude

print("skewness of the Venue Longitude" , skew(venues['Venue Longitude']))

print("kurtosis of the Venue Longitude" , kurtosis(venues['Venue Longitude']))
# Import Libraries

import matplotlib.pyplot as plt

import seaborn as sns
# Histogram

venues['Venue Longitude'].hist()
# Plot

fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(venues['Venue Longitude'])

plt.show()
# Boxplot

sns.boxplot(venues['Venue Longitude'])
# Histogram

venues['Venue Latitude'].hist()
# Plot

fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(venues['Venue Latitude'])

plt.show()
# Boxplot

sns.boxplot(venues['Venue Latitude'])
# Histogram

venues['Venue Name'].hist()
# Plot

fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(venues['Venue Name'])

plt.show()
# Histogram

venues['Venue Category'].hist()
#Plot

fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(venues['Venue Category'])

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='Venue Name',y='Venue Latitude',data=venues)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='Venue Name',y='Venue Longitude',data=venues)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='Venue Category',y='Venue Latitude',data=venues)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='Venue Category',y='Venue Longitude',data=venues)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='Venue Latitude',y='Venue Longitude',data=venues)

plt.show()
sns.distplot(venues['Venue Latitude'])
sns.distplot(venues['Venue Longitude'])
plt.figure(figsize=(14,10))

sns.heatmap(venues.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)

plt.show()
sns.pairplot(data=venues)
# Import libraries

from sklearn.preprocessing import scale

venues1=venues.drop(columns = {'Venue Name' , 'Venue Category'})

venues1
venues_scale=scale(venues1)
venues_scale
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(venues_scale, method='ward'))
# The x-axis contains the samples and y-axis represents the distance between these samples. The vertical line with maximum distance is the blue line and hence we can decide a threshold of 8 and cut the dendrogram:



plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(venues_scale, method='ward'))

plt.axhline(y=8, color='r', linestyle='--')
# We have two clusters as this line cuts the dendrogram at two points. Let’s now apply hierarchical clustering for 2 clusters

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  

cluster.fit_predict(venues_scale)
# Import LIbraries

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
# apply K-means clustering

wss = []

#Sum_of_squared_distances = []

K = np.arange(1,50)

for k in K:  

    km = KMeans(n_clusters=k)

    km = km.fit(venues_scale)

    wss.append(km.inertia_)

plt.plot(K, wss, 'bx-')

plt.xlabel('k')

plt.ylabel('K')

plt.title('Elbow Method For Optimal k')

plt.show()
# Apply K-menas 

model=KMeans(n_clusters=3)

model
# Fit the model

venues_fit = model.fit(venues_scale)

venues_fit
# Predict the model

from sklearn.cluster import KMeans

y_kmeans = model.predict(venues_scale)

y_kmeans
# Add the new column to existing

venues['clusters']=pd.Series(y_kmeans)

venues