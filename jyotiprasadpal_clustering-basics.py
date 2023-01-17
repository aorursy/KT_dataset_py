import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans, AgglomerativeClustering



from scipy.cluster.hierarchy import dendrogram,linkage
# We have a dataframe:

# Step-1 Check the scale of the columns - use StandardScaler, MinMaxScaler to convvert them in same scale

# Step-2  Find out number of clusters - optimum clusters - elbow plot = 3

# Step 3 = # Use Sklearn to run K means clustering

#from sklearn.cluster import KMeans



#kmeans_obj = KMeans(n_clusters= 3)

#kmeans_obj.fit(demo_df)



# Step4 - Do the cluster labelling for all the  datapoints - this will help u to understand which datapoint falls in which cluster.
demo_df = pd.DataFrame({'Salary':[10,15,18,20,32,37,39,45,42,65,66,68,72],

                        'CreditCardBill':[4,6,8,10,14,15,18,20,25,32,36,38,38]

                       })
mms = MinMaxScaler()

demo_df_scaled = mms.fit_transform(demo_df)
demo_df_scaled
demo_df.head()
plt.scatter(demo_df['Salary'], demo_df['CreditCardBill'],color='k')
# Understand: when we will have more than two columns we will not be able to visualize 
# Use Sklearn to run K means clustering

kmeans_obj = KMeans(n_clusters= 3)

kmeans_obj.fit(demo_df)
# centers of the clusters

clus_centers = kmeans_obj.cluster_centers_

clus_centers
# Cluster labels tells u which datapoints fall in which cluster

cluster_lables = kmeans_obj.predict(demo_df)

cluster_lables
color_combinations = {1:'b',2:'r',3:'g',4:'m',5:'y'}

clus_colors = map(lambda x:color_combinations[x+1], cluster_lables)



# fig, ax = plt.subplots(figsize=(6, 4))

# ax.set_title('Within cluster distance with change of value of k')

# ax.set_xlabel('Number of clusters - K')

# ax.set_ylabel('Within cluster distance')

# sns.pairplot(data=demo_df, kind='scatter', vars=['Salary', 'CreditCardBill'])



plt.scatter(demo_df['Salary'], demo_df['CreditCardBill'], color = list(clus_colors))
# Assuming we do not know number of clusters - what are the directional ( Subjective ) ways to get

# a guideline on what should be number of cluster. One option is it use elblow plot.
cdist(demo_df, kmeans_obj.cluster_centers_, metric = 'euclidean')
np.min(cdist(demo_df,kmeans_obj.cluster_centers_, metric = 'euclidean'),axis=1)
within_cluster_dist = []

k_range = range(1,10)

for k in k_range:

    kmeans_model = KMeans(n_clusters=k)

    kmeans_model.fit(demo_df)

    within_cluster_dist.append(sum(np.min(cdist(demo_df,kmeans_model.cluster_centers_,metric = 'euclidean'),axis=1)))
within_cluster_dist
fig, ax = plt.subplots(figsize=(6, 4))

ax.set_title('Within cluster distance with change of value of k')

ax.set_xlabel('Number of clusters - K')

ax.set_ylabel('Within cluster distance')

sns.lineplot(x=k_range, y=within_cluster_dist, ax=ax)
demo_df.head()
# Calculation of distance

linkages_in_hclust = linkage(demo_df.values,'single')



# Drawing of dendrogram ( passing the distance)

dendrogram(linkages_in_hclust,

          distance_sort='descending')
# RULE TO DEFINE NUMBER OF CLUSTERS FROM CLUSTER DENDROGRAM IS -

# TAKE THE LONGEST VERTICAL WHICH IS NOT CUT BY ANY HORIZONTAL LINE 

# AND THEN CUT THE LINE THROUGH A HORIZONATAL - THE RESULTANT VERTICA
# DEndrogram gives us no.of clusters.

# Things to note - Try creating propotionate clusters - each clusters should have similar proportion of data points
hcluster = AgglomerativeClustering(n_clusters=3)

hcluster.fit_predict(demo_df)
# Mean based - K MEANS CLUSTERING - Exclusive Clustering

# Parent Child - Hierarchical Clustering - Exclusive ( Parent -Child overlap)



# Both these approach uses euclidean distance

# CON of Hierarchical clustering :



# Kmeans distance calculations vs Hierarchical distance calculation

# For large dataset it becomes infeasible in Hierarchical clustering because of the number of distance calclation required 
# C means clustering = Fuzzy clustering = K means overlapping clustering
# Methods of finding out number of clusters



# HELPING METHOD - combination of methods

# elbow method

# Dendrogram

# Looking at the cluster split (CLUSTERS HAS TO MAKE SENSE TO THE PEOPLE WHO ARE USING IT)

#(Deployment of clustering is rare - Anomaly dteection , fraud detection)