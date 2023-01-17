import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from scipy.stats import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from scipy.stats import jarque_bera
from scipy.stats import normaltest

import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import time

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import DBSCAN

from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

title_font={"family": "arial", "weight": "bold", "color": "darkred", "size": 12}
label_font={"family": "arial", "weight": "bold", "color": "darkblue", "size": 10}
df= pd.read_csv("../input/creating-customer-segments/customers.csv")
df.head()
df.head().T
df.info() #There is 440 rows and 8 columns on data. However 'Channel' and 'Region' columns must be nominal. 
df["Channel"].replace({1:"h_r_c", 2:"Retail"}, inplace=True)    #Channel:{Hotel/Restaurant/Cafe - 1, Retail - 2}
df["Region"].replace({1:"Lisbon", 2:"Porto", 3:"Other"}, inplace=True)  #Region:{Lisbon - 1, Oporto - 2, or Other - 3} (Nominal)
df.head() #Looks nice!
df.isnull().sum()*100/df.shape[0] #No null value appears.
plt.figure(figsize=(12,8))
col_names= ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
for i in range(6):
    plt.subplot(2,3,i+1)
    ax=sns.boxplot(x=df[col_names[i]], linewidth=2.5)
    plt.title(col_names[i], fontdict=title_font)
plt.show()

#There is appear some outliers.
plt.figure(figsize=(16,8))
col_names= ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
for i in range(6):
    plt.subplot(3,2,i+1)
    sns.distplot(df[col_names[i]], kde=False)
    plt.title(col_names[i], fontdict=title_font)
plt.show()
df["log_Fresh"]= np.log(df["Fresh"])
df["log_Milk"]= np.log(df["Milk"])
df["log_Grocery"]= np.log(df["Grocery"])
df["log_Frozen"]= np.log(df["Frozen"])
df["log_Detergents_Paper"]= np.log(df["Detergents_Paper"])
df["log_Delicatessen"]= np.log(df["Delicatessen"])

plt.figure(figsize=(14,8))
col_names_log= ["Fresh", "log_Fresh", "Milk", "log_Milk", "Grocery", "log_Grocery",
                "Frozen", "log_Frozen", "Detergents_Paper", "log_Detergents_Paper",
                "Delicatessen", "log_Delicatessen"]
for i in range(12):
    plt.subplot(6,2,i+1)
    sns.distplot(df[col_names_log[i]], kde=False)
    plt.title(col_names_log[i], fontdict=title_font)
plt.show()
    
#logarithmic transformation may have worked for get rid of outliers
col_names= ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
col_names_log= ["log_Fresh", "log_Milk", "log_Grocery", "log_Frozen", "log_Detergents_Paper", "log_Delicatessen"]
for col in col_names:
    comparison= pd.DataFrame(columns={"col_name", "threshold", "outliers", "outliers_log"})
    q75, q25= np.percentile(df[col],[75,25])
    q75_log, q25_log= np.percentile(np.log(df[col]), [75,25])
    caa= q75-q25
    caa_log= q75_log,q25_log 
    for threshold in np.arange(0,5,0.5):
        min_value= q25-(caa*threshold)
        max_value= q75+(caa*threshold)
        min_value_log= q25_log-(caa*threshold)
        max_value_log= q75_log+(caa*threshold)
        outliers= len(np.where((df[col]>max_value) | (df[col]<min_value))[0])
        outliers_log= len(np.where((np.log(df[col])>max_value) | (np.log(df[col])<min_value))[0])
        comparison= comparison.append({"col_name":col, "threshold":threshold, "outliers":outliers, "outliers_log":outliers_log},
                                           ignore_index=True)
    display(comparison)
    
    #As we see all outliers are running out with logarithmic transformation. 
    #After that we keep going with logarithmic transformation of columns
for col in col_names:
    df[col]= np.log(df[col])
for col_log in col_names_log:
    del df[col_log]
    
df.columns
df.describe()
plt.figure(figsize=(18,6))
sns.countplot(data=df, x=df["Region"], hue=df["Channel"], palette="Blues_d", order=["Lisbon", "Porto", "Other"])
plt.title("REGİON", fontdict=title_font)
plt.xlabel("Region", fontdict=label_font)
plt.ylabel("Count", fontdict=label_font)
plt.tick_params(colors="black")
plt.show()

df.corr()
correlation_matrix=df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(correlation_matrix, square=True, annot=True, vmin=0, vmax=1, linewidth=0.5)
plt.title("Correlation Matrix Of Continuous Features", fontdict=title_font)
plt.show()

#Only a relationship appears between Fresh and Milk columns.
df= pd.concat([df, pd.get_dummies(df["Channel"], drop_first=True), pd.get_dummies(df["Region"], drop_first=True)], axis=1)
df.drop(columns=["Channel", "Region"], axis=1, inplace=True)
df.head()  #We have transformed categorical columns to dummy.
# Before clustering, we transform features from original version to standardize version
scaler= StandardScaler()
df_std= scaler.fit_transform(df)
#Creating model
k_means= KMeans(n_clusters=3, random_state=123) 
%timeit k_means.fit(df_std) # ""%timeit" shows the running time of the model
clustering= k_means.predict(df_std)

#plot cluster size
plt.hist(clustering)
plt.title("Sales Per Cluster", fontdict=title_font)
plt.xlabel("Clusters", fontdict=label_font)
plt.ylabel("Sales", fontdict=label_font)
plt.show()
#Trying model with different n_clusters and compare the silhouette scores 
silhouette= pd.DataFrame(columns={"n_clusters", "silhouette_score"})
for n_cluster in range(2,20):
    k_means= KMeans(n_clusters=n_cluster, random_state=123).fit_predict(df_std)
    s_scores= metrics.silhouette_score(df_std, k_means, metric="euclidean")
    silhouette= silhouette.append({"n_clusters": n_cluster, "silhouette_score": s_scores}, ignore_index=True)
silhouette.sort_values(by="silhouette_score", ascending=False)
#plot scores
s_scores=[]
for n_cluster in range(2,15):
    s_scores.append(metrics.silhouette_score(df_std, KMeans(n_clusters=n_cluster).fit_predict(df_std), metric="euclidean"))
    
n= [2,3,4,5,6,7,8,9,10,11,12,13,14]
sns.barplot(x=n, y=s_scores, palette="Blues_d")
plt.title("Comparison The Silhouette Scores", fontdict=title_font)
plt.xlabel("Number of Clusters", fontdict=label_font)
plt.ylabel("Silhouette Scores", fontdict=label_font)
plt.show()
agg_cluster= AgglomerativeClustering(n_clusters=3, affinity="euclidean").fit_predict(df_std)

plt.hist(agg_cluster)
plt.title("Sales Per Cluster", fontdict=title_font)
plt.xlabel("Clusters", fontdict=label_font)
plt.ylabel("Sales", fontdict=label_font)
plt.show()
#Trying model with different parameters
link_df= pd.DataFrame(columns={"n_clusters", "linkage", "silhouette_score"})
for n_cluster in range(2,15):
    for link in ["ward","complete","average"]:
        agg_cluster= AgglomerativeClustering(n_clusters=n_cluster, affinity="euclidean", linkage=link).fit_predict(df_std)
        s_scores= metrics.silhouette_score(df_std, agg_cluster, metric="euclidean")
        link_df= link_df.append({"n_clusters": n_cluster, "linkage":link, "silhouette_score":s_scores}, ignore_index=True)
link_df= link_df.sort_values(by="silhouette_score", ascending=False)
display(link_df)
plt.figure(figsize=(10,7))
sns.barplot(x=link_df.iloc[0:10,2], y=link_df.iloc[0:10,1], palette="Blues_d", hue=link_df["linkage"])
plt.title("Sales Per Cluster", fontdict=title_font)
plt.xlabel("Clusters", fontdict=label_font)
plt.ylabel("Silhouette Score", fontdict=label_font)
plt.show()
# Creating model with different parameters
dbscan_df= pd.DataFrame(columns={"eps", "min_samples", "silhouette_score", "n_clusters"})
for eps in np.arange(0.8,2,0.1):
    for min_sample in range(1,10): 
        dbscan_clusters= DBSCAN(eps=eps, min_samples=min_sample).fit(df_std)
        s_scores=  metrics.silhouette_score(df_std, dbscan_clusters.labels_, metric='euclidean')
        dbscan_df= dbscan_df.append({"eps":eps, "min_samples":min_sample, "silhouette_score":s_scores,
                                     "n_clusters":len(set(dbscan_clusters.labels_))}, ignore_index=True)
dbscan_df= dbscan_df.sort_values(by="silhouette_score", ascending=False)
display(dbscan_df.head())
gmm_cluster= GaussianMixture(n_components=2, random_state=123).fit_predict(df_std)

plt.hist(gmm_cluster)
plt.title("Sales Per Cluster", fontdict=title_font)
plt.xlabel("Clusters", fontdict=label_font)
plt.ylabel("Sales", fontdict=label_font)
plt.show()
# Trying model with different parameters
gmm_df= pd.DataFrame(columns={"n_components", "silhouette_score"})
for n in range(2,10):
    gmm_cluster= GaussianMixture(n_components=n, random_state=123).fit_predict(df_std)
    s_scores=metrics.silhouette_score(df_std, gmm_cluster, metric="euclidean")
    gmm_df= gmm_df.append({"n_components":n, "silhouette_score":s_scores}, ignore_index=True)
gmm_df= gmm_df.sort_values(by="silhouette_score", ascending=False)
display(gmm_df)
k_means_cluster= KMeans(n_clusters=3, random_state=123).fit_predict(df_std)
hierarchy_cluster= AgglomerativeClustering(n_clusters=None, affinity="euclidean", distance_threshold=4.2, linkage="average").fit_predict(df_std)
dbscan_cluster= DBSCAN(eps=1.9, min_samples=3).fit(df_std)
gmm_cluster= GaussianMixture(n_components=2).fit_predict(df_std)

k_means_s= metrics.silhouette_score(df_std, k_means_cluster, metric="euclidean")
hierarchy_s= metrics.silhouette_score(df_std, hierarchy_cluster, metric="euclidean")
dbscan_s= metrics.silhouette_score(df_std, dbscan_cluster.labels_, metric="euclidean")
gmm_s= metrics.silhouette_score(df_std, gmm_cluster, metric="euclidean")

compare= [["KMeans", k_means_s], ["Hierarchical", hierarchy_s], ["DBSCAN", dbscan_s], ["GMM", gmm_s]]
compare= pd.DataFrame(compare, columns=["Clustering Method", "Silhouette Score"])
compare= compare.sort_values(by="Silhouette Score", ascending=False)
display(compare)

df["cluster"]=hierarchy_cluster
df["cluster"].value_counts() #If we use n_clusters=2, as seems clusters are not suitable. 
                             #Thats why we did not use n_clusters=2. We used distance_threshold=4.2 parameter. 
plt.figure(figsize=(20,12))
threshold_list= [3.6, 3.9, 4.2, 4.5, 4.8, 5.1]
for i in range(6):
        plt.subplot(3,2,i+1)
        dendrogram(linkage(df_std, method="average"), color_threshold=threshold_list[i])
        plt.title("Dendrogram for threshold:= {}".format(threshold_list[i]), fontdict=title_font)
        plt.xlabel("Sample Index or Cluster Size", fontdict=label_font)
        plt.ylabel("Distance", fontdict=label_font)
plt.show()
hierarchy_cluster= AgglomerativeClustering(n_clusters=None, affinity="euclidean", distance_threshold=4.2,
                                            linkage="average").fit_predict(df_std)
df["cluster"]=hierarchy_cluster
df["cluster"].value_counts()  #We can use 3 cluster that names "1", "2", and "other" cluster
df["cluster"]= k_means_cluster  #It looks good.
df["cluster"].value_counts()
df_norm= sklearn.preprocessing.normalize(df_std)
df_norm.shape
pca= PCA(n_components=7).fit(df_norm)
pca_all= pca.fit_transform(df_norm)

print("Original sahpe of data: {}".format(df_norm.shape))
print("Transformed shape of data: {}".format(pca_all.shape))
np.cumsum(pca.explained_variance_ratio_)  #Explained variance scores.
plt.plot(np.cumsum(pca.explained_variance_ratio_))
tsne= TSNE(n_components=2, perplexity=40, n_iter=300).fit(df_std)
tsne2= tsne.fit_transform(df_std)
print("Original shape of data: {}".format(df_std.shape))
print("Transformed shape of data: {}".format(tsne2.shape))
df_tsne= pd.DataFrame(tsne2, columns={"D1", "D2"})
df_tsne.head()
plt.figure(figsize=(12,6))
plt.scatter(df_tsne["D1"], df_tsne["D2"], c=KMeans(n_clusters=3, random_state=123).fit_predict(tsne2), cmap="viridis" )
plt.show()
tsne= TSNE(n_components=3, perplexity=40, n_iter=300).fit(df_std)
tsne3= tsne.fit_transform(df_std)
print("Original shape of data: {}".format(df_std.shape))
print("Transformed shape of data: {}".format(tsne3.shape))
df_tsne3= pd.DataFrame(tsne3, columns={"D1", "D2", "D3"})
df_tsne3.head()
fig= px.scatter_3d(df_tsne3, x=df_tsne3["D1"], y=df_tsne3["D2"], z=df_tsne3["D3"],
                   color=KMeans(n_clusters=3, random_state=123).fit_predict(tsne3))
fig.show()
umap_fit= umap.UMAP(n_neighbors=5, min_dist=0.3,metric="correlation")
result_umap= umap_fit.fit_transform(df_norm)
print("Original shape of data: {}".format(df_norm.shape))
print("Transformed shape of data: {}".format(result_umap.shape))
df_umap= pd.DataFrame(result_umap, columns={"D1", "D2"})
df_umap.head()
plt.figure(figsize=(12,6))
plt.scatter(df_umap["D1"], df_umap["D2"], c=KMeans(n_clusters=3, random_state=123).fit_predict(result_umap), cmap="viridis")
plt.show()
kmeans3= KMeans(n_clusters=3, random_state=123).fit(df_std)
df["cluster"]=kmeans3.labels_
columns= ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper',
           'Delicatessen', 'h_r_c', 'Other', 'Porto']
for col in columns:
    plt.figure(figsize=(14,3))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        cluster= df[df["cluster"]==i]
        cluster[col].hist()
        plt.title("{} \n {}".format(col, i), fontdict=title_font)
    plt.tight_layout()
    plt.show()
        
        


fig= px.scatter_3d(df, x=df["Fresh"], y=df["Milk"], z=df["Grocery"], color=df["cluster"], title="Fresh - Milk - Grocery")
fig1= px.scatter_3d(df, x=df["Frozen"], y=df["Detergents_Paper"], z=df["Delicatessen"], color=df["cluster"],
                    title="Frozen - Detergents And Paper - Delicatessen")
fig2= px.scatter_3d(df, x=df["h_r_c"], y=df["Other"], z=df["Porto"], color=df["cluster"], title="h_r_c - Porto - Other")

fig.show()
fig1.show()
fig2.show()