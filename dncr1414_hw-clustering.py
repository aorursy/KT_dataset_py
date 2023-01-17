# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart.csv")
data.head(10)
data["target"].unique()
data.info()
import matplotlib.pyplot as plt
plt.scatter(data['age'],data['chol'])

plt.xlabel("age")

plt.ylabel("chol")

plt.show()
#KMeans Clustering

data2=data.loc[:,["age","chol"]]



from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3)

kmeans.fit(data2)



labels=kmeans.predict(data2)

plt.scatter(data["age"],data["chol"],c=labels)

plt.xlabel("age")

plt.ylabel("chol")

plt.show()
#inertia

inertia_list=np.empty(8)

for i in range(1,8):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i]=kmeans.inertia_



plt.plot(range(0,8),inertia_list)

plt.xlabel("Number of cluster")

plt.ylabel("İnertia")

plt.show()
data3=data.drop("target",axis=1)
#Hierarchy

from scipy.cluster.hierarchy import linkage,dendrogram



merg=linkage(data3.iloc[200:220,:],method="single")

dendrogram(merg,leaf_rotation=90,leaf_font_size=6)

plt.show()