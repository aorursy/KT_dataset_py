import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

import sklearn.utils
data = pd.read_csv('/kaggle/input/country-socioeconomic-data/Country-data.csv')

data.head(10)
data.describe()
data.info()
print(data.shape)
df = data.iloc[:, 1:data.shape[1]]

df
# checking the missing values

data.isnull().sum()
data.columns
# checking for outliers using box-plots



plt.figure(figsize=(20,20), dpi=100)



plt.subplot(4,3,1)

sns.boxplot(x = 'child_mort', data = data)



plt.subplot(4,3,2)

sns.boxplot(x = 'exports', data = data)



plt.subplot(4,3,3)

sns.boxplot(x = 'health', data = data)



plt.subplot(4,3,4)

sns.boxplot(x = 'imports', data = data)



plt.subplot(4,3,5)

sns.boxplot(x = 'income', data = data)



plt.subplot(4,3,6)

sns.boxplot(x = 'inflation', data = data)



plt.subplot(4,3,7)

sns.boxplot(x = 'life_expec', data = data)



plt.subplot(4,3,8)

sns.boxplot(x = 'total_fer', data = data)



plt.subplot(4,3,9)

sns.boxplot(x = 'gdpp', data = data)


# checking for outliers using the Z-score



from scipy import stats



z = np.abs(stats.zscore(data[['child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp']]))

print(z)

print("*******************************************************************************")



# threshold = 3  # selecting 3 as the threshold to identify outliers

print('Below are the outlier points along with the respective column numbers in the second array')

print("\n")

print(np.where(z > 3))
data_outliers_removed = data[(z<3).all(axis=1)]

data_outliers_removed.head()

print('Shape of dataframe before outlier removal: ' + str(data.shape))

print("\n")

print('Shape of dataframe after outlier removal: ' + str(data_outliers_removed.shape))
num_vars = data_outliers_removed.drop('country',axis =1)  

cat_vars = data_outliers_removed['country']
num_vars
# Standardization of the dataset before performing PCA



scaler = StandardScaler()

num_vars_scaled = scaler.fit_transform(num_vars)

num_vars_scaled_df = pd.DataFrame(num_vars_scaled,columns=num_vars.columns)

num_vars_scaled_df
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))       

sns.heatmap(num_vars_scaled_df.corr(),annot = True)
pca = PCA(random_state=42)

pca.fit(num_vars_scaled)

var_cumu = np.cumsum(pca.explained_variance_ratio_)

fig = plt.figure(figsize=[12,8],dpi=80)

plt.vlines(x=4, ymax=1, ymin=0, colors="r", linestyles="--")

plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")

plt.plot(var_cumu)

plt.ylabel("Cumulative variance explained")

plt.show()
pca_final = IncrementalPCA(n_components=4)

num_vars_pca_final = pca_final.fit_transform(num_vars_scaled)

print(num_vars.shape)

print(num_vars_pca_final.shape)
# Finding the Optimal Number of Clusters

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)

    kmeans.fit(num_vars_pca_final)

    

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters

# ssd

plt.plot(ssd)
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)

    kmeans.fit(num_vars_pca_final)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(num_vars_pca_final, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
# final model with k=3

kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=42)

kmeans.fit(num_vars_pca_final)
data_outliers_removed['K-Means_Cluster_ID'] = kmeans.labels_
healthy_df_clus_temp = df[["child_mort","income","gdpp",]]

healthy_df_clus_temp = StandardScaler().fit_transform(healthy_df_clus_temp)

db = DBSCAN(eps = 0.3, min_samples = 3).fit(healthy_df_clus_temp)

dataFrameLabels = pd.DataFrame(db.labels_)

dataFrameLabels.head(15)
# Building the label to colour mapping 

colours = {}

colours[0]  = 'red' # Zero values on labels

colours[1]  = 'green'# Minus 1 values on labels

colours[-1]  = 'blue'# Two values on labels
cvec = [colours[label] for label in dataFrameLabels]

colors = ['red', 'green', 'blue'] 

print(cvec)

set(cvec)
#Visualize Results

# For the construction of the legend of the plot 

colors = ['r','g','b']

r = plt.scatter( df["child_mort"],df["gdpp"], marker ='o', color =colours[0] ) 

g = plt.scatter( df["income"],df["gdpp"], marker ='o', color = colours[1]) 

b = plt.scatter( df["income"],df["child_mort"], marker ='o', color = colours[-1]) 

plt.legend((r, g, b), 

           ('Label 0', 'Label 2', 'Label -1'), 

           scatterpoints = 1, 

           loc ='upper right', 

           ncol = 3, 

           fontsize = 8) 
num_vars_scaled_df.head()
# complete linkage

cl_mergings = linkage(num_vars_scaled_df, method="complete", metric='euclidean')

dendrogram(cl_mergings)

plt.show()
# 4 clusters using complete linkage

cl_cluster_labels = cut_tree(cl_mergings, n_clusters=4).reshape(-1, )

cl_cluster_labels
data_outliers_removed["Hierarchical_Cluster_labels"] = cl_cluster_labels

data_outliers_removed.head()

# plotting sub-plots to analyse the results



plt.figure(figsize=(16,14), dpi=100)



plt.subplot(3,2,1)

sns.boxplot(x='K-Means_Cluster_ID', y='gdpp', data=data_outliers_removed)



plt.subplot(3,2,2)

sns.boxplot(x='Hierarchical_Cluster_labels', y='gdpp', data=data_outliers_removed)



plt.subplot(3,2,3)

sns.boxplot(x='K-Means_Cluster_ID', y='child_mort', data=data_outliers_removed)



plt.subplot(3,2,4)

sns.boxplot(x='Hierarchical_Cluster_labels', y='child_mort', data=data_outliers_removed)



plt.subplot(3,2,5)

sns.boxplot(x='K-Means_Cluster_ID', y='income', data=data_outliers_removed)



plt.subplot(3,2,6)

sns.boxplot(x='Hierarchical_Cluster_labels', y='income', data=data_outliers_removed)
# scatter plot using the gdpp, child_mort to observe the cluster distribution



plt.figure(figsize=(15,6),dpi=90)



plt.subplot(1,2,1)

sns.scatterplot(x='gdpp',y='child_mort',data=data_outliers_removed,hue='K-Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='gdpp',y='child_mort',data=data_outliers_removed,hue='Hierarchical_Cluster_labels')



plt.figure(figsize=(15,6),dpi=90)



plt.subplot(1,2,1)

sns.scatterplot(x='gdpp',y='income',data=data_outliers_removed,hue='K-Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='gdpp',y='income',data=data_outliers_removed,hue='Hierarchical_Cluster_labels')
plt.figure(figsize=(15,6),dpi=90)



plt.subplot(1,2,1)

sns.scatterplot(x='child_mort',y='income',data=data_outliers_removed,hue='K-Means_Cluster_ID')



plt.subplot(1,2,2)

sns.scatterplot(x='child_mort',y='income',data=data_outliers_removed,hue='Hierarchical_Cluster_labels')
K_Means_countries = data_outliers_removed[data_outliers_removed['K-Means_Cluster_ID'] == 1]

K_Means_countries.head(10)
Hirarchical_countries = data_outliers_removed[data_outliers_removed['Hierarchical_Cluster_labels'] == 0]

Hirarchical_countries.head(10)
# countries common to both the models



common_countries = pd.merge(K_Means_countries,Hirarchical_countries,how='inner',on=['country', 'child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp', 'K-Means_Cluster_ID',

       'Hierarchical_Cluster_labels'])

common_countries
common_countries[['country', 'child_mort', 'income','gdpp']].head(10)
# dataframe with dereasing child mortality rate and increasing income



common_countries_final = common_countries[['country', 'child_mort','income','gdpp']].sort_values(['child_mort','income'],ascending=[False,True])

common_countries_final.head(15)
final_countries = common_countries_final[(common_countries_final['child_mort'] > 80) &  (common_countries_final['income'] < 1200)]

final_countries = final_countries.reset_index(drop=True)

final_countries