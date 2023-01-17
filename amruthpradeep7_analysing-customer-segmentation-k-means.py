# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing libs

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Importing the dataset

dataset = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#Inpsecting the data

dataset.head()
dataset.info()

#From the information we can see that there are 4 integer variables and 1 categorical variable.

dataset.isnull().sum()

#There is no missing data.
dataset.shape
#Pairplot

sns.pairplot(dataset)
#Checking correlation from heatmap

sns.heatmap(dataset.corr(), annot=True,cmap='Greens')

#This heatmap shows most correlated feature in dark green color and less correlated feature in light green color.
#Feature Selection for the model

X = dataset.iloc[:, 3:].values
#Finding the K no. of clusters by using elbow method

from sklearn.cluster import KMeans

clusters = []



for i in range(1,11):

    km = KMeans(n_clusters=i)

    km.fit(X)

    clusters.append(km.inertia_)

    

plt.plot(range(1,11),clusters)

plt.title('Elbow point')

plt.xlabel('no of clusters')

plt.ylabel('inertia')

plt.show()    
#Modelling

km = KMeans(n_clusters=5, random_state=0)

y_means = km.fit_predict(X)
#Plotting the clusters

plt.scatter(X[y_means == 0,0], X[y_means == 0,1], s=100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_means == 1,0], X[y_means == 1,1], s=100, c= 'blue', label = 'Cluster 2')

plt.scatter(X[y_means == 2,0], X[y_means == 2,1], s=100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_means == 3,0], X[y_means == 3,1], s=100, c='cyan', label = 'Cluster 4')

plt.scatter(X[y_means == 4,0], X[y_means == 4,1], s=100, c='magenta', label = 'Cluster 5')



plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='purple', label = 'centroids')



plt.title('Cluster of customers')

plt.xlabel('Income')

plt.ylabel('Score')

plt.legend()

plt.show()