import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_excel("../input/east-west-airlines/EastWestAirlines.xlsx", sheet_name='data')
dataset.head()
# Column rename.



dataset= dataset.rename(columns={'ID#':'ID', 'Award?':'Award'})
# not going to falloe EDA step here since it is already done in link1.(Above cell)

# as we know ID & award will not make much contribution during clutering. we will drop both columns.



dataset1 =  dataset.drop(['ID','Award'], axis=1)

dataset1.head(2)


from sklearn.preprocessing import StandardScaler



std_df = StandardScaler().fit_transform(dataset1)      # this will used for kmeans

std_df.shape
# Using Minmaxscaler for accuracy result comparison



from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()



minmax_df = minmax.fit_transform(dataset1)

minmax_df.shape
# applying PCA on std_df



# we are considering 95% variance in n_components to not loose any data.



from sklearn.decomposition import PCA

pca_std = PCA(random_state=10, n_components=0.95)

pca_std_df= pca_std.fit_transform(std_df)
# eigenvalues..



print(pca_std.singular_values_)
# variance containing in each formed PCA



print(pca_std.explained_variance_ratio_*100)
# Cummulative variance ratio..



# this will give an idea of, at how many no. of PCAs, the cummulative addition of

#........variance will give much information..



cum_variance = np.cumsum(pca_std.explained_variance_ratio_*100)

cum_variance
# applying PCA on minmax_df



from sklearn.decomposition import PCA



pca_minmax =  PCA(random_state=10, n_components=0.95)

pca_minmax_df = pca_minmax.fit_transform(minmax_df)
# eigenvalues..



print(pca_minmax.singular_values_)
# variance containing in each formed PCA



print(pca_minmax.explained_variance_ratio_*100)
# 1. How many number of clusters? n_clusters?



# Since true labels are not known..we will Silhouette Coefficient (Clustering performance evaluation)

# knee Elbow graph method



#Import the KElbowVisualizer method

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer





# Instantiate a scikit-learn K-Means model. we will check for two diff hyperparameters value effect.

model1 = KMeans(random_state=0,n_jobs=-1,)

model2 = KMeans(random_state=10, n_jobs=-1, max_iter=500, n_init=20,)



# Instantiate the KElbowVisualizer with the number of clusters and the metric

visualizer1 = KElbowVisualizer(model1, k=(2,10), metric='silhouette', timings=False)

visualizer2 = KElbowVisualizer(model2, k=(2,10), metric='silhouette', timings=False)

# Fit the data and visualize

print('model1')

visualizer1.fit(pca_std_df)    

visualizer1.poof()

plt.show()



print('model2')

visualizer2.fit(pca_std_df)    

visualizer2.poof()

plt.show()
from sklearn.metrics import silhouette_score



list1= [2,3,4,5,6,7,8,9]  # always start number from 2.



for n_clusters in list1:

    clusterer1 = KMeans(n_clusters=n_clusters, random_state=0,n_jobs=-1)

    cluster_labels1 = clusterer1.fit_predict(pca_std_df)

    sil_score1= silhouette_score(pca_std_df, cluster_labels1)

    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score1)
# 1. How many number of clusters? n_clusters?



# Since true labels are not known..we will Silhouette Coefficient (Clustering performance evaluation)

# knee Elbow graph method



#Import the KElbowVisualizer method

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer





# Instantiate a scikit-learn K-Means model. we will check for two diff hyperparameters value effect.

model3 = KMeans(random_state=0,n_jobs=-1)

model4 = KMeans(random_state=10, n_jobs=-1, max_iter=500, n_init=20)



# Instantiate the KElbowVisualizer with the number of clusters and the metric

visualizer3 = KElbowVisualizer(model3, k=(2,10), metric='silhouette', timings=False)

visualizer4 = KElbowVisualizer(model4, k=(2,10), metric='silhouette', timings=False)

# Fit the data and visualize

print('model3')

visualizer3.fit(pca_minmax_df)    

visualizer3.poof()

plt.show()



print('model4')

visualizer4.fit(pca_minmax_df)    

visualizer4.poof()

plt.show()
from sklearn.metrics import silhouette_score



list1= [2,3,4,5,6,7,8,9]  # always start number from 2.



for n_clusters in list1:

    clusterer2 = KMeans(n_clusters=n_clusters, random_state=0,n_jobs=-1)

    cluster_labels2 = clusterer1.fit_predict(pca_minmax_df)

    sil_score2= silhouette_score(pca_std_df, cluster_labels2)

    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score2)
# we have found good number of cluster = 6

# model building using cluster numbers = 6



model1 = KMeans(n_clusters=6, random_state=0,n_jobs=-1)

y_predict1 = model1.fit_predict(pca_std_df)

y_predict1.shape
# these are nothing but cluster labels...



y_predict1
# y_predict & cluster labels both are same use any one of them to avoid further confusion.



model1.labels_
# cluster centres associated with each lables



model1.cluster_centers_
# within-cluster sum of squared



# The lower values of inertia are better and zero is optimal.

# Inertia is the sum of squared error for each cluster. 

# Therefore the smaller the inertia the denser the cluster(closer together all the points are)



model1.inertia_
model1.score(pca_std_df) 



# it is opposite value of sum of squared value..avoid to use it. It is bit confusing
# this will give what hyper parameter is used in model.





model1.get_params()
from yellowbrick.cluster import SilhouetteVisualizer



fig,(ax1,ax2) = plt.subplots(1,2,sharey=False)

fig.set_size_inches(15,6)







sil_visualizer1 = SilhouetteVisualizer(model1,ax= ax1, colors=['#922B21','#5B2C6F','#1B4F72','#32a84a','#a83232','#323aa8'])

sil_visualizer1.fit(pca_std_df)





# 2nd Plot showing the actual clusters formed



import matplotlib.cm as cm

colors1 = cm.nipy_spectral(model1.labels_.astype(float) / 6) # 6 is number of clusters

ax2.scatter(pca_std_df[:, 0], pca_std_df[:, 1], marker='.', s=30, lw=0, alpha=0.7,

                c=colors1, edgecolor='k')



# Labeling the clusters

centers1 = model1.cluster_centers_

# Draw white circles at cluster centers

ax2.scatter(centers1[:, 0], centers1[:, 1], marker='o',c="white", alpha=1, s=200, edgecolor='k')



for i, c in enumerate(centers1):

    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')





ax2.set_title(label ="The visualization of the clustered data.")

ax2.set_xlabel("Feature space for the 1st feature")

ax2.set_ylabel("Feature space for the 2nd feature")



plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "

                  "with n_clusters = %d" % 6),fontsize=14, fontweight='bold')



sil_visualizer1.show()

plt.show()

# Creating dataframe of cluster lables..



model1_cluster = pd.DataFrame(model1.labels_.copy(), columns=['Kmeans_Clustering'])
# Concating model1_Cluster df with main dataset copy



Kmeans_df = pd.concat([dataset.copy(), model1_cluster], axis=1)

Kmeans_df.head()
# Plotting barplot using groupby method to get visualize how many row no. in each cluster



fig, ax = plt.subplots(figsize=(10, 6))

Kmeans_df.groupby(['Kmeans_Clustering']).count()['ID'].plot(kind='bar')

plt.ylabel('ID Counts')

plt.title('Kmeans Clustering (pca_std_df)',fontsize='large',fontweight='bold')

ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')

ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.show()
# Applying Dendrogram on PCA data. Or you may apply it on Standardized/normalized indepedent variable data.

# Here diffrent linkage method from hyperparameter is used to see diff between methods for understanding. 

# Ward method is commanly used since it is simpler to visualize understanding.

# Find number of cluster's using color coding of dendrogram. Each color indicates one cluster.



# import scipy.cluster.hierarchy as shc

# for methods in ['single','complete','average','weighted','centroid','median','ward']: 

#     plt.figure(figsize =(20, 6)) 

    

#     dict = {'fontsize':24,'fontweight' :16, 'color' : 'blue'}

    

#     plt.title('Visualising the data, Method- {}'.format(methods),fontdict = dict) 

#     Dendrogram1 = shc.dendrogram(shc.linkage(pca_std_df, method = methods,optimal_ordering=False))

    

# Note: the execution of this cell takes time so i have attached output graphs below
from sklearn.cluster import AgglomerativeClustering

n_clusters = [2,3,4,5,6,7,8]  # always start number from 2.



for n_clusters in n_clusters:

    for linkages in ["ward", "complete", "average", "single"]:

        hie_cluster1 = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkages) # bydefault it takes linkage 'ward'

        hie_labels1 = hie_cluster1.fit_predict(pca_std_df)

        silhouette_score1 = silhouette_score(pca_std_df, hie_labels1)

        print("For n_clusters =", n_clusters,"The average silhouette_score with linkage-",linkages, ':',silhouette_score1)

    print()

 ### Here i have avoded to apply dendrogram since it takes time to run code.
from sklearn.cluster import AgglomerativeClustering

n_clusters = [2,3,4,5,6,7,8]  # always start number from 2.



for n_clusters in n_clusters:

    for linkages in ["ward", "complete", "average", "single"]:

        hie_cluster2 = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkages) # bydefault it takes linkage 'ward'

        hie_labels2 = hie_cluster2.fit_predict(pca_minmax_df)

        silhouette_score2 = silhouette_score(pca_minmax_df, hie_labels2)

        print("For n_clusters =", n_clusters,"The average silhouette_score with linkage-",linkages, ':',silhouette_score2)

    print()
agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='average')

y_pred_hie = agg_clustering.fit_predict(pca_std_df)

print(y_pred_hie.shape)

y_pred_hie
# Cluster numbers



agg_clustering.n_clusters_
# cluster labels for each point



agg_clustering.labels_
# Number of leaves in the hierarchical tree.



agg_clustering.n_leaves_
# The estimated number of connected components in the graph.



agg_clustering.n_connected_components_
# The children of each non-leaf node. Values less than n_samples correspond to leaves of 

#......the tree which are the original samples. A node i greater than or equal to n_samples 

#.........is a non-leaf node and has children children_[i - n_samples]. Alternatively at the 

#...........i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i



agg_clustering.children_
# Clustering Score



(silhouette_score(pca_std_df, agg_clustering.labels_)*100).round(3)
# Plotting Dendrogram.



import scipy.cluster.hierarchy as shc

for methods in ['average']: 

    plt.figure(figsize =(20, 6)) 

    

    dict = {'fontsize':24,'fontweight' :16, 'color' : 'blue'}

    

    plt.title('Visualising the data, Method- {}'.format(methods),fontdict = dict) 

    Dendrogram2 = shc.dendrogram(shc.linkage(pca_std_df, method = methods,optimal_ordering=False))
# Creating dataframe of cluster lables..



hie_cluster = pd.DataFrame(agg_clustering.labels_.copy(), columns=['Hie_Clustering'])
# Concating model1_Cluster df with main dataset copy



hie_df = pd.concat([dataset.copy(), hie_cluster], axis=1)

hie_df .head()
# Plotting barplot using groupby method to get visualize how many row no. in each cluster



fig, ax = plt.subplots(figsize=(10, 6))

hie_df.groupby(['Hie_Clustering']).count()['ID'].plot(kind='bar')

plt.ylabel('ID Counts')

plt.title('Hierarchical Clustering (pca_std_df)',fontsize='large',fontweight='bold')

ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')

ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.show()
Kmeans_df.groupby(['Kmeans_Clustering']).count()
hie_df.groupby(['Hie_Clustering']).count()
# Groupby Cluster lables



count_df = Kmeans_df.groupby(['Kmeans_Clustering']).count()

count_df
# Total numbers in each cluster..



count = count_df.xs('ID' ,axis = 1)

count.plot(kind='bar', title= 'Nuber Counts')

plt.show()
# Sorting elements based on cluster label assigned and taking average for insights.



cluster1 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==0].mean(),columns= ['Cluster1_avg'])

cluster2 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==1].mean(),columns= ['Cluster2_avg'])

cluster3 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==2].mean(),columns= ['Cluster3_avg'])

cluster4 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==3].mean(),columns= ['Cluster4_avg'])

cluster5 = pd.DataFrame(Kmeans_df.loc[Kmeans_df.Kmeans_Clustering==4].mean(),columns= ['Cluster5_avg'])
avg_df = pd.concat([cluster1,cluster2,cluster3,cluster4,cluster5],axis=1)

avg_df
# Extract and plot one Column data .xs method

for i , row in avg_df.iterrows():

    fig = plt.subplots(figsize=(8,6))

    j = avg_df.xs(i ,axis = 0)

    plt.title(i, fontsize=16, fontweight=20)

    j.plot(kind='bar',fontsize=14)

    plt.show()

    print()