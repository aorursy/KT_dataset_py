import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.cluster import KMeans
df=pd.read_csv('../input/usarrests/USArrests.csv')
df.head()
df.index
df.size
df.shape
# stil during 'Unnamed: 0' let's get out of the list
df.columns
# Let's see if I have any missing observations
df.isnull().sum()
df.rename( columns={'Unnamed: 0':'States'}, inplace=True )
df.rename( columns={'Rape':'Rape_rate'}, inplace=True )
df.rename( columns={'Murder':'Murder_rate'}, inplace=True )

# data.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )
df.columns
df.set_index('States',inplace=True)
df.describe()
df.index
df.loc['North Carolina','Murder_rate']
df.columns = list(map(str, df.columns))
df.columns
df.index
df.dtypes
df[['Assault', 'UrbanPop']] = df[['Assault', 'UrbanPop']].astype('float')
df.dtypes
df = df/np.max(df)
df.describe().T
# here it is appropriate to visualize the data to try to understand it.
# For example, 3 Assault (Assault) histogram looks like 3 peaks or accumulation.
df.hist(figsize = (10,10));
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans
df.set_index(['States'],inplace =True)
k_fit = kmeans.fit(df)
k_fit.n_clusters
k_fit.cluster_centers_
# If I want to visualize, now let's reduce the set numbers to 2
kmeans = KMeans(n_clusters = 2)
k_fit = kmeans.fit(df)        
sets = k_fit.labels_
# Let's visualize the data we reduced to 2 sets.

plt.scatter(df.iloc[:,0], df.iloc[:,1], c = sets, s = 50, cmap = "viridis")

centers = k_fit.cluster_centers_                                 # We want to create 2 centers and show them on the visual.

plt.scatter(centers[:,0], centers[:,1], c = "black", s = 200, alpha = 0.5);
# Let us import 3D visualization. Otherwise it is necessary to download

from mpl_toolkits.mplot3d import Axes3D


# Let's create our sets again, this time it will be 3 dimensional variable

kmeans = KMeans(n_clusters = 3)
k_fit = kmeans.fit(df)
sets = k_fit.labels_
centers = kmeans.cluster_centers_
plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]);
# Let's look at the visualization of these sets and centers on the plot.

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=sets)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
           marker='*', 
           c='#050505', 
           s=1000);
# To provide cluster numbers and information about which states (observations) these numbers belong to
# If we want, we can take the model with 2 variables or 3 variables above, let's take the 3 ones

kmeans = KMeans(n_clusters = 3)
k_fit = kmeans.fit(df)
sets = k_fit.labels_
# to see which set and index you have for the top 10 states

pd.DataFrame({"Provinces" : df.index, "Sets": sets})[0:15]
# to look at the set number that each belongs to

df["set_no"] = sets

df.head()
df["set_no"] = df["set_no"] + 1

df.head()
# It is called one from 2 to 50. The number of sets should decrease, because we should approach zero, because we reduce the sands.
# When you have 10 thousand customers, you are not interested in 100 people. It is necessary to put the customers with high degrees or features into a segment(sets).
#!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,50))
visualizer.fit(df) 
visualizer.poof()  

# We understand the presentation from gorsel each point segment (set), ie, the set of elements with similar properties in it
# For example, when the customer enters our site, a presentation can be made about what the monthly income it brings to us.
# Let's take our model above again
# To provide cluster numbers and information about which states (observations) these numbers belong to
# If we want we can take the model with 2 variables or 3 variables above, let's take the 4 normal ones

kmeans = KMeans(n_clusters = 4)
k_fit = kmeans.fit(df)
sets = k_fit.labels_
# to see which set and index you have for the top 10 states

pd.DataFrame({"Provinces" : df.index, "Sets": sets})[0:10]
df = pd.read_csv("/Users/raufsafarov/Downloads/US_violent_crime.csv").copy()
df.index = df.iloc[:,0]
df = df.iloc[:,1:5]
#del df.index.name
df.index.name = "Index"
df.head()
from scipy.cluster.hierarchy import linkage

hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")
hc_single = linkage(df, "single")
# We can watch its features and see what it does

dir(hc_complete)
# We need to create Dendogram

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('Hierarchical Clustering - Dendogram')
plt.xlabel('Indexs')
plt.ylabel('Distance')
dendrogram(
    hc_complete,
    leaf_font_size=10
);
# another form of representation and the number of elements below it

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('Hierarchical Clustering - Dendogram')
plt.xlabel('Indexs')
plt.ylabel('Distance')
dendrogram(
    hc_complete,
    truncate_mode = "lastp",
    p = 4,
    show_contracted = True
);
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('Hierarchical Clustering - Dendogram')
plt.xlabel('Indexs')
plt.ylabel('Distance')
den = dendrogram(
    hc_complete,
    leaf_font_size=10
);
# When we look at the dendogram, it will be logical to divide it into 4 large clusters. Then we say n_cluster = 4.

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 4, 
                                  affinity = "euclidean", 
                                  linkage = "ward")

cluster.fit_predict(df)
# if we want to see which state is in which bank

pd.DataFrame({"Provinces" : df.index, "Sets": cluster.fit_predict(df)})[0:10]
df["set_no"] = cluster.fit_predict(df)
df.head()













