import pandas as pd

df = pd.read_csv('../input/Police_Incidents.csv',low_memory=False)

print ("Number of Incidents: ", len(df))

print ("Number of Columns: ", df.shape[1])
pd.options.display.max_columns = 103

df.head()
#confirm header names; be careful of spaces

list(df)
import numpy as np

df2 = df[np.isfinite(df[' X Coordinate '])]
div = df.groupby(by='Division').count()

div
#assigning each division a number for color mapping

def color(row):

    if row['Division'] == 'Central':

        return 1

    elif row['Division'] == 'North Central':

        return '2' 

    elif row['Division'] == 'NorthEast':

        return '3'  

    elif row['Division'] == 'NorthWest':

        return '4'  

    elif row['Division'] == 'South Central':

        return '5'  

    elif row['Division'] == 'SouthEast':

        return '6' 

    else:

        return '7' 



df['DivisionColor'] = df.apply(color, axis=1)



from matplotlib import pyplot as plt

%matplotlib inline

plt.figure(figsize=(12, 8))

plt.scatter(df[' X Coordinate '],df[' Y Coordinate'], c=df['DivisionColor'], cmap=plt.cm.rainbow, s=35, linewidths=1, alpha=.3)

plt.xlabel('Long'), plt.ylabel('Lat')

plt.grid()
#first let's see what seven clusters looks like, in order to replicate their divisions

from sklearn.cluster import KMeans



X1 = df2[[' X Coordinate ',' Y Coordinate']]



cls = KMeans(n_clusters=7, init='random')

cls.fit(X1)

newfeature_fare = cls.labels_ # the labels from kmeans clustering



X1=X1.values

plt.figure(figsize=(12, 8))

plt.scatter(X1[:, 1], X1[:, 0], c=newfeature_fare,  cmap=plt.cm.rainbow, s=35, linewidths=1, alpha=.3)

plt.xlabel('Long'), plt.ylabel('Lat')

plt.grid()



print ("Cluster Centers Are:")

print ("")

print (cls.cluster_centers_)

print ("")

print ("Sum of Distances of Each Dot to Cluster Center:",)

print ("")

print (cls.inertia_)
#is seven too many? let's see what five clusters looks like.

X1 = df2[[' X Coordinate ',' Y Coordinate']]



cls = KMeans(n_clusters=5, init='random')

cls.fit(X1)

newfeature_fare = cls.labels_ # the labels from kmeans clustering to be applied as color in scatterplot



X1=X1.values

plt.figure(figsize=(12, 8))

plt.scatter(X1[:, 1], X1[:, 0], c=newfeature_fare,  cmap=plt.cm.rainbow, s=35, linewidths=1, alpha=.3)

plt.xlabel('Long'), plt.ylabel('Lat')

plt.grid()



print ("Cluster Centers Are:")

print ("")

print (cls.cluster_centers_)

print ("")

print ("Sum of Distances of Each Dot to Cluster Center:",)

print ("")

print (cls.inertia_)
#is seven too few? let's see what 10 clusters looks like.

X1 = df2[[' X Coordinate ',' Y Coordinate']]



cls = KMeans(n_clusters=10, init='random')

cls.fit(X1)

newfeature_fare = cls.labels_ # the labels from kmeans clustering



X1=X1.values

plt.figure(figsize=(12, 8))

plt.scatter(X1[:, 1], X1[:, 0], c=newfeature_fare,  cmap=plt.cm.rainbow, s=35, linewidths=1, alpha=.3)

plt.xlabel('Long'), plt.ylabel('Lat')

plt.grid()



print ("Cluster Centers Are:")

print ("")

print (cls.cluster_centers_)

print ("")

print ("Sum of Distances of Each Dot to Cluster Center:",)

print ("")

print (cls.inertia_)