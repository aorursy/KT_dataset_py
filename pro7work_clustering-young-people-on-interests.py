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
import matplotlib.pyplot as plt
import seaborn as sns
responses=pd.read_csv("../input/responses.csv")
columns=pd.read_csv("../input/columns.csv")
responses.head()
#creating a hobbies dataframe
hobbies=responses.iloc[:,31:63,]

hobbies.head()
hobbies.dtypes
print("hobbies:",hobbies.shape)

#finding missing values in dataset
#hobbies.isnull().any()
#Checking if any null values in the columns
hobbies.loc[hobbies.isnull().any(axis=1)]
#dropping all the Nan values
hobbies.dropna(inplace=True)
hobbies.loc[hobbies.isnull().any(axis=1)]
hobbies.shape

hobbies.describe()
# Finding outliers in Internet and FunWith Friends

#sns.boxplot(x=hobbies['Internet'])
plt.boxplot([hobbies['Internet'],hobbies['Fun with friends']])
from scipy import stats
import numpy as np
outliers=[]

z_score=np.abs(stats.zscore(hobbies))
z_score
#print(np.where(z_score > 3))
zscore_df= pd.DataFrame(hobbies.iloc[[3,8,38,97, 105, 132, 169, 180, 220, 260, 275, 291, 327,
       373, 408, 416, 517, 519, 557, 656, 672, 675, 687, 699, 760, 804,848],:])
# Finding out people  who are not Interested in Internet at all
zscore_df[zscore_df['Internet']==1]
hobbies.drop(index=747, inplace=True)
hobbies.shape
#Finding people with no interest in Socialising
zscore_df[zscore_df['Fun with friends']==2]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
hobbies_scaled=ss.fit_transform(hobbies)
hobbies_scaled= pd.DataFrame(hobbies_scaled)
hobbies_scaled.head(20)
#  clustermap plot using Seaborn


sns.clustermap(hobbies.corr(), center=0, cmap="seismic",
                             linewidths=.75, figsize=(13, 13))
#df= hobbies.T
df=hobbies
# Using the elbow method to find  the optimal number of clusters
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10,random_state=0)
    km=kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.figure()
plt.plot(range(1,11),wcss, marker="o")

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares')

plt.show()
# create a Series for dataframe index and values
hobbies_series=pd.Series(df.index )
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def silhouette_analysis(df, cluster_labels,n,clustering_type,heading):
    
    plt.figure(figsize=(15,10))
    ax= plt.subplot()
    ax.set_ylim([0, len(df) + (n + 1) * 50])
    dictofhobbies={i:cluster_labels[i] for i in range(0,len(cluster_labels))}
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    if clustering_type == 'Kmeans':
        metrics='euclidean'
    if clustering_type == 'Agglomerative':
        metrics='euclidean'
    if clustering_type == 'GaussianMixture':
        metrics='mahalanobis'
    silhouette_avg = silhouette_score(df, cluster_labels,metric=metrics)
    
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    
   
    yticks=[]
    ylabels=[]
    y_lower = 10
    for i in range(n):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        silhouette_labels=[hobby for hobby,cluster_label in dictofhobbies.items() if cluster_label==i ]
        zipped_values=dict(zip(silhouette_labels,ith_cluster_silhouette_values))      
        #sorted(zipped_values,key=lambda x:x1])
        new_zipped_values=sorted(zipped_values.items(), key=lambda x: x[1])
        ith_cluster_silhouette_values.sort()
        #print("ith_cluster_silhouette_values::",len(ith_cluster_silhouette_values))
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        #print("yupper is::",y_upper)
        color = cm.nipy_spectral(float(i) / n)
        pos=np.arange(y_lower, y_upper)
       
        ax.barh(pos,ith_cluster_silhouette_values,height=1.0,color=color, edgecolor="none")
               
        yticks.extend(pos)
        ylabels.extend(labels[0] for labels in new_zipped_values)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
       
        # Compute the new y_lower for next plot
        y_lower = y_upper + 50  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Clusters")
    
    
    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    hobbies_labels=[]
    for y in ylabels:
        hobbies_labels.append(hobbies_series[y])
    
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(hobbies_labels, fontSize=8)  
    ax.set_xticks([-0.1, 0,0.1, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for " ,heading, "with n_clusters = %d" % n),
                 fontsize=14, fontweight='bold')

plt.show()

from sklearn.cluster import KMeans

range_of_clusters=[6,7,8,9]
for n in range_of_clusters:
    kmeans_sil=KMeans(n_clusters=n,max_iter=300,n_init=10,random_state=10)
    cluster_labels=kmeans_sil.fit_predict(df)

    silhouette_analysis(df,cluster_labels,n,'Kmeans','Kmeans')
#using agglomerative clustering
from sklearn.cluster import AgglomerativeClustering

range_of_clusters=[6,7,8,9]
for n in range_of_clusters:
    ag_c= AgglomerativeClustering(n_clusters=n)
    ag_cluster_labels=ag_c.fit_predict(df)
    silhouette_analysis(df,ag_cluster_labels,n,'Agglomerative','Agglomerative')
#also trying the gaussian mixture models

from sklearn.mixture import GaussianMixture
range_of_clusters=[6,7,8,9]
for n in range_of_clusters:
    gmm= GaussianMixture(n_components=n)
    gmm_cluster_labels=gmm.fit_predict(df)
    silhouette_analysis(df,gmm_cluster_labels,n, 'GaussianMixture','GaussianMixture')
#Running PCA on the scaled data


from sklearn.decomposition import PCA
plt.figure()
cmr=[]
for n_com in range(2,20):
    
    cumulative_ratio=0
   
    pca= PCA(n_components=n_com)
    hobbies_reduced=pca.fit_transform(hobbies_scaled)

    print("for components = %d" %n_com)
    components= pd.DataFrame(np.round(pca.components_, 4),columns=df.keys())
   
    ratios=pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    
    for i in range((variance_ratios.shape[0])):
        
        cumulative_ratio= cumulative_ratio + variance_ratios.values[i]
    
    cmr.append(cumulative_ratio)
    print(cumulative_ratio)
vr= variance_ratios.values
plt.plot(range(1,20),vr*100,marker="x")   
plt.xlabel("PCA Components")
plt.ylabel("Explained Variance ratio")
plt.figure()
plt.plot(range(2,20),cmr,marker="o")
plt.xlabel("No. of Components")
plt.ylabel("cumulative Variance ratio")
plt.show()
#running PCA with kmeans 
for n_comp in [15]:
    print("PCA for components=%d"%n_comp)
    pca_km=PCA(n_components=n_comp)
    df_reduced=pca_km.fit_transform(hobbies_scaled)
    
    
    for n_clust in [6]:
        kmeans_red_pca=KMeans(n_clusters=n_clust)
        reduced_cluster_labels=kmeans_red_pca.fit_predict(df_reduced)
        centers_km=kmeans_red_pca.cluster_centers_
        silhouette_analysis(df_reduced,reduced_cluster_labels,n_clust,'Kmeans','Kmeans After PCA for components %d' %n_comp)
        
    

#visualizing the 6 clusters in 2 dimensions with 2 principal componenets
#reducing data to 2 prinicipal components
pca_2comp=PCA(n_components=2)
df_reduced_2comp=pca_2comp.fit_transform(hobbies_scaled)

# running Kmeans for 6 clusters on 2 PC's
kmeans_pca_2comp=KMeans(n_clusters=6)
reduced_cluster_labels_2comp=kmeans_pca_2comp.fit_predict(df_reduced_2comp)
centers_2comp=kmeans_pca_2comp.cluster_centers_

#creating dataframes for cluster labels and 2 PC's
predictions=pd.DataFrame(reduced_cluster_labels_2comp,columns=['Cluster_pred'])
reduced_df_2comp=pd.DataFrame(np.round(df_reduced_2comp,4),columns=['Dimension 1','Dimension 2'])

# concatanete the above two dfs
to_plot=pd.concat([predictions,reduced_df_2comp], axis=1)
to_plot.shape

def visualize_clusters(to_plot,centers,n_clusters):
    
    plt.figure(figsize=(15,8))
    ax=plt.subplot()

    colors=['red','green','blue','orange','yellow','purple']
    for n_clusters in range(n_clusters):
        
        #colors = cm.nipy_spectral((to_plot['Cluster_pred']== n_clusters).astype(float) / n_clusters)
        #colors=['red','green','blue','orange','yellow','purple']
        
        ax.scatter(to_plot[to_plot['Cluster_pred']==n_clusters].iloc[:, 1],to_plot[to_plot['Cluster_pred']==n_clusters].iloc[:, 2],
                   marker='.', s=80, lw=0, alpha=0.7,
                    c=colors[n_clusters], edgecolor='black', label='Cluster %d'%n_clusters)

            
            # Draw white circles at cluster centers
        

        for i, c in enumerate(centers):
           
            ax.scatter(c[0], c[1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
            ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=80, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    plt.legend()    
    plt.show()
# visualize clusters
visualize_clusters(to_plot,centers_2comp,6)
# Lets recover the data reduced by PCA (also Standard scalar for scaled data)

#finding out the real centers of the data
#main_centers=ss.inverse_transform(pca_2comp.inverse_transform(centers_2comp))
real_centers= ss.inverse_transform(pca_km.inverse_transform(centers_km))

real_centers_df=pd.DataFrame(np.round(real_centers,3), columns = df.columns)
#Deviation from the median(as mean is sensitive to outliers)

dev_from_median=real_centers_df- hobbies.median()
dev_from_median
#Plotting the distributions in clusters
for i in range(len(dev_from_median.index)):
    plt.figure(figsize=(10,8))
  
    dev_from_median.iloc[i,:].plot(kind='bar')
    plt.title("Distribution of Interests in Cluster %d"%i)
    plt.grid(axis='x')
    #plt.axhline(y=3.5,color="red", linestyle="--")
    
    plt.show()
   
