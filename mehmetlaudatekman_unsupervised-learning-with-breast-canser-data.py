# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.tail()
data.info()

data2 = data.copy() 

data.drop(["id","Unnamed: 32","diagnosis"],axis=1,inplace=True)

data.head()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2) # I've chosen n_clusters=2 because there are two classes in our dataset.

kmeans_labels = kmeans.fit_predict(data)
from sklearn.cluster import AgglomerativeClustering

hier = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")

hier_labels = hier.fit_predict(data)
data2.drop(["Unnamed: 32","id"],axis=1,inplace=True)

data2["KMeansLabel"] = kmeans_labels

data2["HierLabel"] = hier_labels
data2.head()
data3 = data2.loc[:,["diagnosis","KMeansLabel","HierLabel"]]

data3
print("Diagnosis = M and Cluster 0 Data Dots: ",len(data3[(data3["diagnosis"]=="M") & (data3.KMeansLabel==0) & (data3.HierLabel==0)]))

print("Diagnosis = M and Cluster 1 Data Dots:",len(data3[(data3["diagnosis"]=="M") & (data3.KMeansLabel==1) & (data3.HierLabel==1)]))

fig,ax = plt.subplots(figsize=(11,7))

plt.scatter(data2["radius_mean"][data2.KMeansLabel==0],data2["texture_mean"][data2.KMeansLabel==0],color="Red")

plt.scatter(data2["radius_mean"][data2.KMeansLabel==1],data2["texture_mean"][data2.KMeansLabel==1],color="Green")

plt.show()