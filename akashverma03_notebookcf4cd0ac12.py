# Q. Import the useful Libraries from description (1 mark)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from scipy.stats import zscore
# Q. Load the wine quality dataset and print the head.

wine = pd.read_csv("winequality-red.csv")

wine.head()
# Q. Count the value of Quality and perform pairplot.

qual = wine["quality"].values

print(qual)

sns.pairplot(wine)

plt.show()

# Q. Drop Quality Feature From Dataset and perform pairplot.

drop = wine.drop("quality",axis=1)

sns.pairplot(drop)

plt.show()
# Q. Scale the Dataset (apply z-score) 

wine.apply(zscore)

# Let us check optimal number of clusters-



# Q. Find suitable no of clusters for K means.

ks=[1,2,3,4,5,6]



# Q. Capture the cluster lables.

kmeans=KMeans(n_clusters=2)

# Q. Capture the centroids.

inertia=[]

for k in ks:

    model=KMeans(n_clusters=k)

    model.fit(wine)



# Q. Capture the intertia. 

    inertia.append(model.inertia_)



# Q. Combine the cluster_range and cluster_errors into a dataframe and print it.

inertia = np.array(inertia)/1000

print(inertia)

inertia = pd.Series(inertia)

ks = pd.Series(ks)

wine["Inertia"] = inertia

wine["Cluster"] = ks    
# Q. Perform the Elbow plot. 

plt.plot(ks,inertia,marker=".")

plt.show()
# Q. Set number of clusters. 

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2)

# Q. Fit the input data. 

import pandas as pd

from sklearn.cluster import KMeans

wine = pd.read_csv("winequality-red.csv")

kmeans.fit(wine)
# Q. Perfrom the Centroids and print it. 

KMeans(n_clusters=2).fit_predict(wine)
# Q. Create a new dataframe only for labels and convert it into categorical variable. 

import pandas as pd

datawine = pd.DataFrame(model.labels_,columns=["WineType"])

datawine ["WineType"][wine_new["WineType"]==0]="Red"

datawine ["WineType"][wine_new["WineType"]==3]="White"

datawine 
# Q. Join the label dataframe with the Wine data frame to create s_df_labeled. Note: it could be appended to original dataframe. (1 mark)

s_df_labeled=wine["WineType"]=datawine 

wine_type=wine.append(s_df_labeled)

wine_type
# Q. Groupby s_df_labeled and label data using Groupby function. 

wine_clusters = wine_type.groupby("WineType")

wine_clusters.head()

#wine_clusters = wine_data_attr.groupby(['clusters'])

## Start code here

num_clusters = np.arange(2,10)

results = {}

for size in num_clusters:

    model = KMeans(n_clusters = size).fit(wine)

    predictions = model.predict(wine)

    results = silhouette_score(wine, predictions)

print(results)

## End code here
# Q. Show the final Boxplot 

fig,ax = plt.subplots()

ax.boxplot(results)

plt.show()
## Start code here

sns.clustermap(wine)

plt.show()

## End code here