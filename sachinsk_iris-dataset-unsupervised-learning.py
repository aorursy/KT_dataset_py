# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

import matplotlib.pyplot as plt

plt.style.use('ggplot')
iris=datasets.load_iris()
X= iris.data
# Import KMeans

from sklearn.cluster import KMeans



# Create a KMeans instance with 3 clusters: model

model = KMeans(n_clusters=3)



# Fit model to points

model.fit(X)



# Determine the cluster labels of new_points: labels

labels = model.predict(X)



# Print cluster labels of new_points

print(labels)

xs = X[:,0]

ys = X[:,2]



# Make a scatter plot of xs and ys, using labels to define the colors

plt.scatter(xs,ys,c=labels,alpha=0.5)



# Assign the cluster centers: centroids

centroids = model.cluster_centers_



# Assign the columns of centroids: centroids_x, centroids_y

centroids_x = centroids[:,0]

centroids_y = centroids[:,1]



# Make a scatter plot of centroids_x and centroids_y

plt.scatter(centroids_x,centroids_y,marker='D',s=50)

plt.show()
model.inertia_
ks = range(1, 6)

inertias = []



for k in ks:

    # Create a KMeans instance with k clusters: model

    model=KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(X)

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

# Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()

# Create a KMeans model with 3 clusters: model

model = KMeans(n_clusters=3)



# Use fit_predict to fit model and obtain cluster labels: labels

labels = model.fit_predict(X)



# Create a DataFrame with labels and varieties as columns: df

df = pd.DataFrame({'labels': labels, 'varieties': iris.target})



# Create crosstab: ct

ct = pd.crosstab(df['labels'],df.varieties)



# Display ct

print(ct)
from sklearn.manifold import TSNE

modle=TSNE(learning_rate=100)

t= model.fit_transform(X)

xs= t[:,0]

ys= t[:,1]

plt.scatter(xs,ys,c=iris.target)
from scipy.stats import pearsonr



# Assign the 0th column of grains: width

sepal_length = iris.data[:,0]



# Assign the 1st column of grains: length

petal_length = iris.data[:,2]



# Scatter plot width vs length

plt.scatter(sepal_length,petal_length)

plt.axis('equal')

plt.show()



# Calculate the Pearson correlation

correlation, pvalue = pearsonr(sepal_length,petal_length)



# Display the correlation

print(correlation)

iris.feature_names
# Import PCA

from sklearn.decomposition import PCA



# Create PCA instance: model

model = PCA()



# Apply the fit_transform method of model to grains: pca_features

pca_features = model.fit_transform(iris.data)



# Assign 0th column of pca_features: xs

xs = pca_features[:,0]



# Assign 1st column of pca_features: ys

ys = pca_features[:,2]





# Scatter plot xs vs ys

plt.scatter(xs, ys)

plt.axis('equal')

plt.show()



# Calculate the Pearson correlation of xs and ys

correlation, pvalue = pearsonr(xs, ys)



# Display the correlation

print(correlation)

print(model.n_components_)

features=range(model.n_components_)
plt.bar(features,model.explained_variance_)

plt.xticks(features)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
from mpl_toolkits.mplot3d import Axes3D 

zs= pca_features[:,2]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs,ys,zs)

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt



# Create scaler: scaler

scaler = StandardScaler()



# Create a PCA instance: pca

pca = PCA()



# Create pipeline: pipeline

pipeline = make_pipeline(scaler,pca)



# Fit the pipeline to 'samples'

pipeline.fit(iris.data)



# Plot the explained variances

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()

from sklearn.decomposition import PCA



# Create PCA instance: model

model = PCA(n_components=2)



# Apply the fit_transform method of model to grains: pca_features

model.fit(iris.data)

trans= model.transform(iris.data)

print(trans.shape)

xs= trans[:,0]

ys= trans[:,1]

plt.scatter(xs,ys, c=iris.target)