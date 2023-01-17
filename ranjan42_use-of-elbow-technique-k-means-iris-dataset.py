# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import datasets
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading the iris dataset
iris = datasets.load_iris()

#Checking the dataset
iris.data
print(iris.data.shape)
#Checking the featues
iris.feature_names
#Scaling the data for clustering for better efficiency
x = scale(iris.data)
#checking the target
iris.target
# Doing the clustering 
clustering = KMeans(n_clusters =3,random_state=1)
#Fitting the algorithm
clustering.fit(x)
#Labelling the cluster
clustering.labels_
#Adding the visualisation
import matplotlib.pyplot as plt
%matplotlib inline
#Converting into DataFrame
iris_df = pd.DataFrame(iris.data)
iris_df.columns=['sepal_length','sepal_width','petal_length','petal_width']
y=pd.DataFrame(iris.target)
y.columns=['targets']
y.head()
plt.scatter(x=iris_df.petal_length,y=iris_df.petal_width)
plt.title("The actual dataset")
import numpy as np
color =np.array(['red','blue','green'])
#adding the colors
plt.scatter(x=iris_df.petal_length,y=iris_df.petal_width,c=color[iris.target])
plt.title("The actual dataset")
#After the clustering
#adding the colors
color2=np.array(['green','red','blue'])
plt.scatter(x=iris_df.petal_length,y=iris_df.petal_width,c=color2[clustering.labels_])
plt.title("The dataset post clustering")
#finding distortion for every k- value

# K = 1 to 10,for every k value we calculate distortion,then we plot the line graph between K and Distortion
distortions = []
K = range(1,18)
print(K)

#importing the cdist package
from scipy.spatial.distance import cdist
import math
iris_df.head()
X = iris_df.loc[:,['petal_length','petal_width']]
# Finding the distortion in every set as the min Euclidean distance between two points
for k in K: 
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    distortions.append(sum(np.min(cdist(X,kmeans_model.cluster_centers_,"euclidean"),axis=1))/X.shape[0])
#printing the distortion
distortions
#Plot distortions againt K value
print(K)
# checking for min of K value
print(np.min(distortions))
#Plotting for change in K value
plt.plot(K,distortions,'bx-')
