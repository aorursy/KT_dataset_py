import pandas as pd
import numpy as np
from scipy.io import mmread
import networkx as nx
from io import StringIO
import matplotlib.pyplot as plt
from pylab import *
# chicago.csv contains edge list only, removed the header(node1,node2)
chicago = nx.read_edgelist('chicago.csv', delimiter=',', create_using=nx.Graph())
nx.draw(chicago,with_labels=True)
show()
print(nx.info(chicago))
def degree_distribution(network):
    import collections
    import matplotlib.pyplot as plt
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)     

    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = network.subgraph(sorted(nx.connected_components(network), key=len, reverse=True)[0])
    pos = nx.spring_layout(network)
    plt.axis('off')
    nx.draw_networkx_nodes(network, pos, node_size=20)
    nx.draw_networkx_edges(network, pos, alpha=0.4)

    plt.show()
    
degree_distribution(chicago)
def traversal_measure(network):
    pathlengths = []

    # print("source vertex {target:length, }")
    for v in network.nodes():
        spl = dict(nx.single_source_shortest_path_length(network, v))
    #     print('{} {} '.format(v, spl))
        for p in spl:
            pathlengths.append(spl[p])

    print('')
    print("average shortest path length %s" % (sum(pathlengths) / len(pathlengths)))

    # histogram of path lengths
    dist = {}
    for p in pathlengths:
        if p in dist:
            dist[p] += 1
        else:
            dist[p] = 1

    print('')
    print("length #paths")
    verts = dist.keys()
    for d in sorted(verts):
        print('%s %d' % (d, dist[d]))

    if nx.is_connected(network):
        print("radius: %d" % nx.radius(network))
        print("diameter: %d" % nx.diameter(network))
        print("center: %s" % nx.center(network))
        print("periphery: %s" % nx.periphery(network))
    else:
        print('Network is not connected, hence radius, diameter, center and periphery cannot be computed')
traversal_measure(chicago)
print("density of Chicago Network: %s" % nx.density(chicago))
avg_cluster_G=nx.average_clustering(chicago)
print ('the average clustering coefficients of Chicago network =', avg_cluster_G)
chicago_deg = nx.degree_histogram(chicago)
chicago_deg_sum = [a * b for a, b in zip(chicago_deg, range(0, len(chicago)))]
print('average degree of CHICAGO Network: {}'.format(sum(chicago_deg_sum) / chicago.number_of_nodes()))
# Eccentricity
if nx.is_connected(chicago):
    ecce_chicago=nx.eccentricity(chicago)
print('the eccentricity of node is',ecce_chicago)
# degree centrality
degree_centrality = nx.algorithms.centrality.degree_centrality(chicago)
print ('the degree centraility of each node is', degree_centrality)
##betweenness
between_chicago=nx.algorithms.centrality.betweenness_centrality(chicago)
print ('the betweenness of each node is', between_chicago)
## Clustering Coefficients
cluster_chicago=nx.clustering(chicago)
print ('the clustering coefficients of each node is', cluster_chicago)
# closeness
close_chicago=nx.algorithms.centrality.closeness_centrality(chicago)
print ('the closeness of each node is', close_chicago)
df=pd.DataFrame([degree_centrality,between_chicago,cluster_chicago,close_chicago,ecce_chicago])
df.corrwith(df,axis=1, drop=False, method='pearson')
df=df.T
df.columns=['Degree centrality','betweenness','Clustering coefficient','Closeness','Eccentricity']
df
# noralize the Eccentricity
from sklearn.preprocessing import MinMaxScaler
df[['Eccentricity']]= MinMaxScaler().fit_transform(df[['Eccentricity']])
df
# for each k = 1 : 10 (including 10), cluster data and compute scores
import seaborn as sns
from sklearn.cluster import MeanShift, KMeans, AffinityPropagation, AgglomerativeClustering, FeatureAgglomeration
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, v_measure_score

distance_to_cluster_centre = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k)  
    kmeans.fit(df)
    distance = np.min(kmeans.transform(df),axis=1)
    average_distance = np.mean(distance)
    distance_to_cluster_centre.append(average_distance)

print('\nCluster centres:')
print(kmeans.cluster_centers_)
# Sum of Squares Calculation
from scipy.spatial.distance import cdist, pdist
K = range(1,11)
KM = [KMeans(n_clusters=k).fit(df) for k in K]
centroids = [k.cluster_centers_ for k in KM]
D_k = [cdist(df, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/df.shape[0] for d in dist]

# Total within sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df)**2)/df.shape[0]
bss = tss-wcss
varExplained = bss/tss*100
# Plot the elbow curve
kIdx = 2
plt.figure(figsize=(10,4)) # Set the size of the plot
plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.subplot(1, 2, 2)
plt.plot(K, varExplained, 'b*-')
plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.tight_layout()
from sklearn.metrics import silhouette_score
sil = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k).fit(df)
    labels = kmeans.labels_
    sil.append(silhouette_score(df, labels, metric='euclidean'))
cluster_sil = np.arange(len(sil)) + 1
plt.plot(cluster_sil, sil)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Scores ')
plt.show()
from sklearn.mixture import GaussianMixture
n_components = np.arange(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df)
          for n in n_components]
plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
plt.plot(n_components, [m.aic(df) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
from sklearn import cluster
import sklearn
from sklearn.mixture import GaussianMixture

silhouette_score_values=list()
 
NumberOfClusters=range(2,11)
 
for i in NumberOfClusters:
    
    gmm = GaussianMixture(i, covariance_type='full', random_state=0)
    gmm.fit(df)
    labels= gmm.predict(df)
    print( "Number Of Clusters:",i)
    print ("Silhouette score value",sklearn.metrics.silhouette_score(df,labels ,metric='euclidean', sample_size=None, random_state=None))
    silhouette_score_values.append(sklearn.metrics.silhouette_score(df,labels ,metric='euclidean', sample_size=None, random_state=None))
 
plt.plot(NumberOfClusters, silhouette_score_values)
plt.title("Silhouette score values vs Numbers of Clusters ")
plt.show()
 
Optimal_NumberOf_Components=NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
print ("Optimal number of components is:")
print (Optimal_NumberOf_Components)
# Produce dendrogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df, method='ward'))
# Apply Hierarchical Clustering for 2 clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(df)
# add the cluster label to the dataset
df['cluster_label'] = pd.Series(cluster.fit_predict(df), index=df.index)
df
df.to_csv('clustered_labels.csv',index=False)
# Visulize the clusters
plt.figure(figsize=(10, 7))  
plt.scatter(df['Eccentricity'], df['betweenness'], c=cluster.labels_) 
import os
import pandas as pd
import numpy as np
import networkx as nx
nyc = nx.read_edgelist('edgeNYC.txt')
# network_nyc

nodes= nyc.nodes
edges=nyc.edges
no_nodes=nyc.number_of_nodes()
no_edges=nyc.number_of_edges()

print(nx.info(nyc))
##eccentricity
if nx.is_connected(nyc):
    ecce_nyc=nx.eccentricity(nyc)
else:
    print('Network is not connected. For a disconnected network, all nodes are defined to have infinite eccentricity.')

##betweenness
between_nyc=nx.algorithms.centrality.betweenness_centrality(nyc)

## degree
deg_nyc=nx.algorithms.centrality.degree_centrality(nyc)

## Closeness
close_nyc=nx.algorithms.centrality.closeness_centrality(nyc)

cluster_nyc=nx.clustering(nyc)
df=pd.DataFrame([between_nyc,cluster_nyc,deg_nyc,close_nyc,ecce_nyc])
df.corrwith(df,axis=1, drop=False, method='pearson')
df=df.T
df.columns=['betweeness','Clustering coefficient','Degree','Closeness','eccentricity']
df.head()
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#prepare dataset for clustering
scaler = MinMaxScaler()
df_scaled=scaler.fit_transform(df)
from sklearn.cluster import KMeans

Num_clusters=range(1,21)
distance_to_cluster_centre = []


for k in Num_clusters:
    kmeans = KMeans(n_clusters=k,random_state=42)  
    kmeans.fit(df_scaled)
    distance = np.min(kmeans.transform(df_scaled),axis=1)
    average_distance = np.mean(distance)
    distance_to_cluster_centre.append(average_distance)

import matplotlib.pyplot as plt
plt.plot(Num_clusters,distance_to_cluster_centre)
plt.title('Elbow Method')
plt.xlabel('Number of Segments')
plt.xticks(Num_clusters)
plt.ylabel('Mean of within cluster distances')
plt.show()


#barplot
plt.bar(Num_clusters, distance_to_cluster_centre)
plt.xlabel('Number of Segments')
plt.ylabel('Mean of within cluster distances')
plt.xticks(Num_clusters)
plt.show()
#cluster_4
kmeans = KMeans(n_clusters=4, random_state=42) 
kmeans.fit(df_scaled)

##storing results
output_kmeans = pd.DataFrame(zip(nodes, kmeans.labels_), 
               columns =['node', 'label']) 
output_kmeans.head()

df['labels']=kmeans.labels_
df.groupby('labels').count()
##Summary of Clusters - average
df['labels']=kmeans.labels_
df.groupby('labels').count()
##Summary of Clusters
cluster_centers = kmeans.cluster_centers_
labels_unique = np.unique(kmeans.labels_)
km_clusters = pd.DataFrame(cluster_centers, columns=['betweeness','Clustering coefficient','Degree','Closeness','eccentricity'])
km_clusters
import seaborn as sns
plt.figure(figsize=(20,10))
sns.set(font_scale=1.5)
sns.heatmap(
    data=km_clusters,
    cmap='Purples',
    annot=True
)
plt.ylabel("KMeansLabel")
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Genre Dendograms")
dend = shc.dendrogram(shc.linkage(df_scaled, method='ward'))
plt.axhline(y=3.5, color='r', linestyle='--')
from sklearn.cluster import AgglomerativeClustering
AC = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit(df_scaled)
##storing results
output_ac = pd.DataFrame(zip(nodes, AC.labels_), 
               columns =['node', 'label']) 
output_ac.head()
from sklearn.mixture import GaussianMixture as GMM


n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full').fit(df_scaled)
          for n in n_components]

plt.plot(n_components, [m.bic(df_scaled) for m in models], label='BIC')
plt.plot(n_components, [m.aic(df_scaled) for m in models], label='AIC')
plt.legend(loc='best')
plt.xticks(n_components)
plt.xlabel('n_components')
gmm = GMM(n_components=4).fit(df_scaled)
labels = gmm.predict(df_scaled)
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=labels, s=40, cmap='viridis')
