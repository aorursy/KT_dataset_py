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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/heart-disease-uci/heart.csv")

data.info()
data.head()
data2 = data.loc[:,["trestbps","chol"]]
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=2)

kmeans.fit(data2)

label = kmeans.predict(data2)



plt.figure(figsize=(5,5))

plt.scatter(data["trestbps"],data["chol"],c=label)
df = pd.DataFrame({"label":label,"target":data["target"]})

ctab = pd.crosstab(df.label,df.target)

print(ctab)
inertia_list = []

for i in range(1,20):

    kmeans = KMeans(n_clusters= i)

    kmeans.fit(data2)

    inertia_list.append(kmeans.inertia_)

    

plt.figure(figsize=(5,5))

plt.plot(range(1,20),inertia_list,"-o")

plt.xticks(range(1,20))

plt.grid()

plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram



a = linkage(data, method="ward")

dendrogram(a,leaf_rotation=90)

plt.show()
from sklearn.cluster import AgglomerativeClustering

hierarchial_cluster = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")

cluster = hierarchial_cluster.fit_predict(data)



data["label"]= cluster

plt.scatter(data.trestbps[data.label == 0 ],data.chol[data.label == 0],color = "red")

plt.scatter(data.trestbps[data.label == 1 ],data.chol[data.label == 1],color = "green")

plt.show()