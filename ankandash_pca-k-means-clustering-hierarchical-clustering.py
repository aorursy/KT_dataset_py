# importing the required libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# loading the required dataset



country_data = pd.read_csv('/kaggle/input/country-socioeconomic-data/Country-data.csv')
country_data.head()
country_data.info()
country_data.describe()
# Checking the percentage of missing values

round(100*(country_data.isnull().sum()/len(country_data.index)), 2)
country_data.columns
# checking for outliers using box-plots



plt.figure(figsize=(20,20), dpi=200)



plt.subplot(4,3,1)

sns.boxplot(x = 'child_mort', data = country_data)



plt.subplot(4,3,2)

sns.boxplot(x = 'exports', data = country_data)



plt.subplot(4,3,3)

sns.boxplot(x = 'health', data = country_data)



plt.subplot(4,3,4)

sns.boxplot(x = 'imports', data = country_data)



plt.subplot(4,3,5)

sns.boxplot(x = 'income', data = country_data)



plt.subplot(4,3,6)

sns.boxplot(x = 'inflation', data = country_data)



plt.subplot(4,3,7)

sns.boxplot(x = 'life_expec', data = country_data)



plt.subplot(4,3,8)

sns.boxplot(x = 'total_fer', data = country_data)



plt.subplot(4,3,9)

sns.boxplot(x = 'gdpp', data = country_data)
# checking for outliers using the Z-score



from scipy import stats



z = np.abs(stats.zscore(country_data[['child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp']]))

print(z)



print("\n")

print("*******************************************************************************")

print("\n")



# threshold = 3  # selecting 3 as the threshold to identify outliers

print('Below are the outlier points along with the respective column numbers in the second array')

print("\n")

print(np.where(z > 3))
# Removing the outliers



country_data_outliers_removed = country_data[(z<3).all(axis=1)]
country_data_outliers_removed.head()
print('Shape of dataframe before outlier removal: ' + str(country_data.shape))

print("\n")

print('Shape of dataframe after outlier removal: ' + str(country_data_outliers_removed.shape))
X = country_data_outliers_removed.drop('country',axis =1)  

y = country_data_outliers_removed['country']
X.shape    
y.shape
X.head()
y.head()
# Standardization of the dataset before performing PCA



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:5,:5]
X.columns
X_scaled_df = pd.DataFrame(X_scaled,columns=X.columns)
X_scaled_df.head()
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))       

sns.heatmap(X_scaled_df.corr(),annot = True)
# we can see that some columns have significant correlation among themselves 
from sklearn.decomposition import PCA
pca = PCA(random_state=42)
pca.fit(X_scaled)
pca.components_[0]
pca.explained_variance_ratio_
var_cumu = np.cumsum(pca.explained_variance_ratio_)

var_cumu
fig = plt.figure(figsize=[12,8],dpi=200)

plt.vlines(x=4, ymax=1, ymin=0, colors="r", linestyles="--")

plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")

plt.plot(var_cumu)

plt.ylabel("Cumulative variance explained")

plt.show()
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=4)
X_pca_final = pca_final.fit_transform(X_scaled)
print(X.shape)

print(X_pca_final.shape)
corrmat = np.corrcoef(X_pca_final.transpose())
corrmat.shape
# Plotting the heatmap of the corr matrix

plt.figure(figsize=[10,10])

sns.heatmap(corrmat, annot=True)
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
# k-means with some arbitrary k (number of clusters)

kmeans = KMeans(n_clusters=5, max_iter=1000)

kmeans.fit(X_pca_final)
kmeans.labels_
# elbow-curve/SSD

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)

    kmeans.fit(X_pca_final)

    

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters

# ssd

plt.plot(ssd)
# from the elbow method we can say that k = 4 clusters seems to be a good choice
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)

    kmeans.fit(X_pca_final)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(X_pca_final, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    
# final model with k=3

kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=42)

kmeans.fit(X_pca_final)
kmeans.labels_
country_data_outliers_removed['K-Means_Cluster_ID'] = kmeans.labels_
country_data_outliers_removed.head()
plt.figure(figsize=(8,4),dpi=200)

sns.boxplot(x='K-Means_Cluster_ID', y='gdpp', data=country_data_outliers_removed)
plt.figure(figsize=(8,4),dpi=200)

sns.boxplot(x='K-Means_Cluster_ID', y='child_mort', data=country_data_outliers_removed)
plt.figure(figsize=(8,4),dpi=200)

sns.boxplot(x='K-Means_Cluster_ID', y='income', data=country_data_outliers_removed)
X_scaled_df.head()
# single linkage

sl_mergings = linkage(X_scaled_df, method="single", metric='euclidean')

dendrogram(sl_mergings)

plt.show()
# complete linkage

cl_mergings = linkage(X_scaled_df, method="complete", metric='euclidean')

dendrogram(cl_mergings)

plt.show()
# 4 clusters using single linkge

sl_cluster_labels = cut_tree(sl_mergings, n_clusters=4).reshape(-1, )

sl_cluster_labels
# the single linkage clustering does not perform well in generating the clusters hence we go for complete linkage
# 4 clusters using complete linkage

cl_cluster_labels = cut_tree(cl_mergings, n_clusters=4).reshape(-1, )

cl_cluster_labels
country_data_outliers_removed["Hierarchical_Cluster_labels"] = cl_cluster_labels
country_data_outliers_removed.head()
plt.figure(figsize=(8,4),dpi=200)

sns.boxplot(x='Hierarchical_Cluster_labels', y='gdpp', data=country_data_outliers_removed)
plt.figure(figsize=(8,4),dpi=200)

sns.boxplot(x='Hierarchical_Cluster_labels', y='child_mort', data=country_data_outliers_removed)
plt.figure(figsize=(8,4),dpi=200)

sns.boxplot(x='Hierarchical_Cluster_labels', y='income', data=country_data_outliers_removed)
country_data_outliers_removed.head()
# plotting sub-plots to analyse the results



plt.figure(figsize=(20,20), dpi=200)



plt.subplot(3,2,1)

sns.boxplot(x='K-Means_Cluster_ID', y='gdpp', data=country_data_outliers_removed)



plt.subplot(3,2,2)

sns.boxplot(x='Hierarchical_Cluster_labels', y='gdpp', data=country_data_outliers_removed)



plt.subplot(3,2,3)

sns.boxplot(x='K-Means_Cluster_ID', y='child_mort', data=country_data_outliers_removed)



plt.subplot(3,2,4)

sns.boxplot(x='Hierarchical_Cluster_labels', y='child_mort', data=country_data_outliers_removed)



plt.subplot(3,2,5)

sns.boxplot(x='K-Means_Cluster_ID', y='income', data=country_data_outliers_removed)



plt.subplot(3,2,6)

sns.boxplot(x='Hierarchical_Cluster_labels', y='income', data=country_data_outliers_removed)
X_pca_final_df = pd.DataFrame(X_pca_final,columns=['PC1','PC2','PC3','PC4'])
X_pca_final_df.head()
X_pca_final_df['K_Means_Cluster_ID'] = kmeans.labels_

X_pca_final_df['Hierarchical_Cluster_Labels'] = cl_cluster_labels
X_pca_final_df.head()
# scatter plot using the first two principal components to observe the cluster distribution



plt.figure(figsize=(12,6),dpi=200)



plt.subplot(1,2,1)

sns.scatterplot(x='PC1',y='PC2',data=X_pca_final_df,hue='K_Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='PC1',y='PC2',data=X_pca_final_df,hue='Hierarchical_Cluster_Labels')
# scatter plot using the gdpp, child_mort to observe the cluster distribution



plt.figure(figsize=(12,6),dpi=200)



plt.subplot(1,2,1)

sns.scatterplot(x='gdpp',y='child_mort',data=country_data_outliers_removed,hue='K-Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='gdpp',y='child_mort',data=country_data_outliers_removed,hue='Hierarchical_Cluster_labels')
# scatter plot using the gdpp, income to observe the cluster distribution



plt.figure(figsize=(12,6),dpi=200)



plt.subplot(1,2,1)

sns.scatterplot(x='gdpp',y='income',data=country_data_outliers_removed,hue='K-Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='gdpp',y='income',data=country_data_outliers_removed,hue='Hierarchical_Cluster_labels')
# scatter plot using the child_mort, income to observe the cluster distribution



plt.figure(figsize=(12,6),dpi=200)



plt.subplot(1,2,1)

sns.scatterplot(x='child_mort',y='income',data=country_data_outliers_removed,hue='K-Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='child_mort',y='income',data=country_data_outliers_removed,hue='Hierarchical_Cluster_labels')
K_Means_countries = country_data_outliers_removed[country_data_outliers_removed['K-Means_Cluster_ID'] == 1]

K_Means_countries
Hirarchical_countries = country_data_outliers_removed[country_data_outliers_removed['Hierarchical_Cluster_labels'] == 0]

Hirarchical_countries
len(Hirarchical_countries)
len(K_Means_countries)
# countries common to both the models



common_countries = pd.merge(K_Means_countries,Hirarchical_countries,how='inner',on=['country', 'child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp', 'K-Means_Cluster_ID',

       'Hierarchical_Cluster_labels'])
common_countries.columns
common_countries[['country', 'child_mort', 'income','gdpp']]
len(common_countries)
# dataframe with dereasing child mortality rate and increasing income

common_countries_final = common_countries[['country', 'child_mort','income','gdpp']].sort_values(['child_mort','income'],ascending=[False,True])

common_countries_final
final_countries = common_countries_final[(common_countries_final['child_mort'] > 80) &  (common_countries_final['income'] < 1200)]

final_countries = final_countries.reset_index(drop=True)

final_countries