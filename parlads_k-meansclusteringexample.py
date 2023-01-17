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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import KMeans
data = pd.read_csv('/kaggle/input/3.01. Country clusters.csv')
data

# data analysis in direction : North and East : positive
#                            : South and west : Negetive   
data['Language'] = data['Language'].map({'English' : 0, 'French' :1 , 'German' : 2})
# plot the data

plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90,90)
plt.show()
# clustering based on location

x= data.iloc[:, 1:4]
x
# clustering
k_means = KMeans(3) #" here KMeans(2) is the number of cluster we want to have in our datasets"
k_means.fit(x)
#clustering result

identified_cluster = k_means.fit_predict(x)
identified_cluster
data_with_cluster = data.copy()
data_with_cluster['Cluster'] =  identified_cluster
data_with_cluster
# map

# plot the data

plt.scatter(data_with_cluster['Longitude'], data_with_cluster['Latitude'], c=data_with_cluster['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90,90)
plt.show()
# wcss 
k_means.inertia_
wcss = []
for i in range(1,7):
    k_means = KMeans(i)
    k_means.fit(x)
    wcss_iter = k_means.inertia_
    wcss.append(wcss_iter)
wcss
# elbow method

number_clusters = range(1,7)

plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')

plt.xlabel('Number of cluster')
plt.ylabel('within cluster sum of squar')

plt.show()