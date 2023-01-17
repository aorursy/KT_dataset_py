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
from warnings import filterwarnings

filterwarnings("ignore")



import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from yellowbrick.cluster import KElbowVisualizer

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
dataset = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
dataset.head()
dataset.isnull().sum()
dataset.shape
# I will use CustomerID as Index of Dataset. Thus I will able to see which clusters include which customers are.

dataset.index = dataset.iloc[:,0]

dataset  = dataset.drop(["CustomerID","Gender"],axis=1) #We can use inplace = True
df = dataset.copy()
dataset.head()
#I don't want to see index name.

dataset.index.name = None
dataset.head()
dataset.describe().T
dataset.hist(figsize=(10,10));
kmeans = KMeans(n_clusters=3)

kmeans
#If you want to look that What attributes does kmeans has, you can use the code below.

#?kmeans
k_fit = kmeans.fit(dataset)
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
X_list = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

y_list = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]





#Setting the size of graphs and specifying  grids

f,ax = plt.subplots(3,3,figsize=(20,16))



#Showing the each pair of variables

for x in range(len(X_list)):

    for y in range(len(y_list)):

        ax[x,y].scatter(dataset[X_list[x]], 

                    dataset[y_list[y]], 

                    c = k_fit.labels_,

                    s = 30, 

                    cmap = "viridis")

        

        #Setting the name of X and Y.

        ax[x,y].set_xlabel(X_list[x])

        ax[x,y].set_ylabel(y_list[y])

        

        #Setting the title of graphs.

        ax[x,y].set_title(X_list[x] + "-" + y_list[y])

        

        # Showing the center of clusters

        ax[x,y].scatter(k_fit.cluster_centers_[:,x],k_fit.cluster_centers_[:,y], c="black", s=200, alpha = 0.5)
pd.DataFrame({"CustomerID" : dataset.index, "Clusters" : k_fit.labels_})[110:200]
#Adding a new column called clusters that include clusters we specified above.

dataset["Clusters"] = k_fit.labels_
dataset.sample(50)
#I want to change 0 type cluster to 1 type cluster so I must increase 1 point every cluster type. 

#My new clusters will be (1,2,3) instead of (0,1,2).



dataset["Clusters"] = dataset["Clusters"] + 1

dataset.sample(50)
kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,20))

visualizer.fit(dataset)

visualizer.poof()
#If we determine the clusters in the same amounts as the observations,

#we would expect the distortion score to be close to zero.



kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,200))

visualizer.fit(dataset)

visualizer.poof()
kmeans = KMeans(n_clusters=6)

k_fit = kmeans.fit(dataset)

clusters =  k_fit.labels_
new_data = pd.DataFrame({"CustormersID" : dataset.index, "Clusters" : clusters})
new_data.sample(20)
#Again, I don't want to see zero as a cluster name.



new_data["Clusters"] = new_data["Clusters"] + 1
new_data.sample(20)
sns.countplot("Clusters", data=new_data)
new_data["Clusters"].value_counts()
hc_complete = linkage(dataset, "complete")

hc_average = linkage(dataset, "average")

hc_single = linkage(dataset, "single")
plt.figure(figsize=(25,10))

plt.title("Hierarchical Cluster - Dendrogram")

plt.xlabel("Index")

plt.ylabel("Distance")



dendrogram(hc_complete,

          leaf_font_size = 10);
# I cannot see the index values so I must truncate the dendrogram.

plt.figure(figsize=(25,10))

plt.title("Hierarchical Cluster - Dendrogram")

plt.xlabel("Index")

plt.ylabel("Distance")



dendrogram(hc_complete,

          truncate_mode = "lastp",

          p = 30,

          show_contracted = True,

          leaf_font_size= 20);
#We have to scale the dataset before PCA implementation.

dataset = StandardScaler().fit_transform(dataset)
dataset[:4]
pca = PCA(n_components=2)

pca_fit = pca.fit_transform(dataset)
pca_df = pd.DataFrame(data=pca_fit,columns=["First_Components","Second_Components"])

pca_df.head(5)
pca.explained_variance_ratio_
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
#One variable can explain around %90 of dataset.