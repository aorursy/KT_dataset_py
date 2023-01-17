'''

We are analyzing IRIS dataset with k-means and hierachical clustering methods



'''
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



%matplotlib inline

from pandas import Series, DataFrame

import pandas as pd

import numpy as np

import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.cluster import KMeans

from pylab import rcParams

rcParams['figure.figsize'] = 9, 8  # set plot size

iris = pd.read_csv("../input/Iris.csv") 

iris.head()
iris_SP = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

iris_SP.head()
iris_SP.describe()
# k-means cluster analysis for 1-15 clusters                                              

from scipy.spatial.distance import cdist

clusters=range(1,15)

meandist=[]



# loop through each cluster and fit the model to the train set

# generate the predicted cluster assingment and append the mean 

# distance my taking the sum divided by the shape

for k in clusters:

    model=KMeans(n_clusters=k)

    model.fit(iris_SP)

    clusassign=model.predict(iris_SP)

    meandist.append(sum(np.min(cdist(iris_SP, model.cluster_centers_, 'euclidean'), axis=1))

    / iris_SP.shape[0])



"""

Plot average distance from observations from the cluster centroid

to use the Elbow Method to identify number of clusters to choose

"""

plt.plot(clusters, meandist)

plt.xlabel('Number of clusters')

plt.ylabel('Average distance')

plt.title('Selecting k with the Elbow Method') 

# pick the fewest number of clusters that reduces the average distance

# If you observe after 3 we can see graph is almost linear
# Here we are just analyzing if we consider 2 cluster instead of 3 by using PCA 

model3=KMeans(n_clusters=2)

model3.fit(iris_SP) # has cluster assingments based on using 2 clusters

clusassign=model3.predict(iris_SP)

# plot clusters

''' Canonical Discriminant Analysis for variable reduction:

1. creates a smaller number of variables

2. linear combination of clustering variables

3. Canonical variables are ordered by proportion of variance accounted for

4. most of the variance will be accounted for in the first few canonical variables

'''

from sklearn.decomposition import PCA # CA from PCA function

pca_2 = PCA(2) # return 2 first canonical variables

plot_columns = pca_2.fit_transform(iris_SP) # fit CA to the train dataset

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,) 

# plot 1st canonical variable on x axis, 2nd on y-axis

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 2 Clusters')

plt.show() 

# close or overlapping clusters idicate correlated variables with low in-class variance 

# but not good separation. 2 cluster might be better.
# calculate full dendrogram

from scipy.cluster.hierarchy import dendrogram, linkage



# generate the linkage matrix

Z = linkage(iris_SP, 'ward')



# set cut-off to 150

max_d = 7.08                # max_d as in max_distance



plt.figure(figsize=(25, 10))

plt.title('Iris Hierarchical Clustering Dendrogram')

plt.xlabel('Species')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',  # show only the last p merged clusters

    p=150,                  # Try changing values of p

    leaf_rotation=90.,      # rotates the x axis labels

    leaf_font_size=8.,      # font size for the x axis labels

)

plt.axhline(y=max_d, c='k')

plt.show()
# calculate full dendrogram for 50

from scipy.cluster.hierarchy import dendrogram, linkage



# generate the linkage matrix

Z = linkage(iris_SP, 'ward')



# set cut-off to 50

max_d = 7.08                # max_d as in max_distance



plt.figure(figsize=(25, 10))

plt.title('Iris Hierarchical Clustering Dendrogram')

plt.xlabel('Species')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',  # show only the last p merged clusters

    p=50,                  # Try changing values of p

    leaf_rotation=90.,      # rotates the x axis labels

    leaf_font_size=8.,      # font size for the x axis labels

)

plt.axhline(y=max_d, c='k')

plt.show()