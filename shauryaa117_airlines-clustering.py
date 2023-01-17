import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import pandas as pd

df = pd.read_csv("../input/AirlinesCluster.csv")
df.head()
df.isnull().sum()
df.info()
for i in df.columns:

    print(i)

    sns.kdeplot(df[i])

    plt.show()

    sns.boxplot(df[i])

    plt.show()
df.describe()
df.head()
plt.figure(figsize=(12,8))

sns.boxplot(data=df)
##Trying out if transformation removes outliers



plt.figure(figsize=(12,8))

sns.boxplot(data=np.sqrt(df))
np.sqrt(df).isnull().sum()
df1 = np.sqrt(df)
df1.head()
df.head()
##Let's use the original data first, and let's remove the outliers from Balance first:
q1 = df['Balance'].quantile(0.25)

q3 = df['Balance'].quantile(0.75)

iqr = q3-q1

ul = q3 + (1.5*iqr)

ll = q1 - (1.5*iqr)

df1 = df[(df['Balance']>ll)&(df['Balance']<ul)]
df1.head()
plt.figure(figsize=(12,8))

sns.boxplot(data=df1)
df.shape
df1.shape
##Now removing outliers from BonusMiles:
q1 = df['BonusMiles'].quantile(0.25)

q3 = df['BonusMiles'].quantile(0.75)

iqr = q3-q1

ul = q3 + (1.5*iqr)

ll = q1 - (1.5*iqr)

df2 = df1[(df1['BonusMiles']>ll)&(df1['BonusMiles']<ul)]
plt.figure(figsize=(12,8))

sns.boxplot(data=df2)
##Removing outliers from FlightMiles:

q1 = df['FlightMiles'].quantile(0.25)

q3 = df['FlightMiles'].quantile(0.75)

iqr = q3-q1

ul = q3 + (1.5*iqr)

ll = q1 - (1.5*iqr)

df3 = df2[(df2['FlightMiles']>ll)&(df2['FlightMiles']<ul)]
plt.figure(figsize=(12,8))

sns.boxplot(data=df3)
##Now removing outliers from QualMiles:

q1 = df['QualMiles'].quantile(0.25)

q3 = df['QualMiles'].quantile(0.75)

iqr = q3-q1

ul = q3 + (1.5*iqr)

ll = q1 - (1.5*iqr)

df4 = df3[(df3['QualMiles']>ll)&(df3['QualMiles']<ul)]
plt.figure(figsize=(12,8))

sns.boxplot(data=df4)
sns.boxplot(df3['QualMiles'])
##We notice that QualMiles contains most outliers, which can be influential while building a model, so we cannot remove them.

#So, Considering Outliers for QualMiles: taking df3 as final
df3.shape
df3.head()
##We need to normalize the data now: 
for i in df3.columns:

    print(i)

    sns.kdeplot(df[i])

    plt.show()
##As we can see above, we need to normalize data:



from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

df_norm = standard_scaler.fit_transform(df3)
from sklearn.cluster import KMeans

cluster_range = range(1,20)

cluster_errors = []

for num_clusters in cluster_range:

    clusters = KMeans(num_clusters,n_init=10)

    clusters.fit(df_norm)

    labels = clusters.labels_

    centroids = clusters.cluster_centers_

    cluster_errors.append(clusters.inertia_)

clusters_df = pd.DataFrame({"num_clusters":cluster_range,"cluster_errors":cluster_errors})

clusters_df[0:20]
#Elbow plot

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(10,8))

plt.plot(clusters_df['num_clusters'],clusters_df['cluster_errors'],marker='o')

plt.xlabel('Num clusters')

plt.ylabel('Cluster Errors')
##4 optimum clusters, as seen from elbow curve:
##Building unsupervised model:
model1 = KMeans(n_clusters = 4, max_iter=50)

model1.fit(df_norm)
#analysis of clusters formed



df3.index = pd.RangeIndex(len(df3.index))

df_km = pd.concat([df3,pd.Series(model1.labels_)],axis=1)

df_km.columns = ['Balance', 'QualMiles', 'BonusMiles', 'BonusTrans', 'FlightMiles',

       'FlightTrans', 'DaysSinceEnroll','ClusterID']
df_km.isna().sum()
df_km
df3.head()
km_cluster_Balance = pd.DataFrame(df_km.groupby('ClusterID')['Balance'].mean())

km_cluster_QualMiles = pd.DataFrame(df_km.groupby('ClusterID')['QualMiles'].mean())

km_cluster_BonusMiles = pd.DataFrame(df_km.groupby('ClusterID')['BonusMiles'].mean())

km_cluster_BonusTrans = pd.DataFrame(df_km.groupby('ClusterID')['BonusTrans'].mean())

km_cluster_FlightMiles = pd.DataFrame(df_km.groupby('ClusterID')['FlightMiles'].mean())

km_cluster_FlightTrans = pd.DataFrame(df_km.groupby('ClusterID')['FlightTrans'].mean())

km_cluster_DaysSinceEnroll = pd.DataFrame(df_km.groupby('ClusterID')['DaysSinceEnroll'].mean())





df = pd.concat([pd.Series([0,1,2,3]),km_cluster_Balance,

km_cluster_QualMiles,

km_cluster_BonusMiles,

km_cluster_BonusTrans,

km_cluster_FlightMiles,

km_cluster_FlightTrans,

km_cluster_DaysSinceEnroll],axis=1)

df.columns = ['ClusterID','Balance', 'QualMiles', 'BonusMiles', 'BonusTrans', 'FlightMiles',

       'FlightTrans', 'DaysSinceEnroll']

df
sns.barplot(data=df,x='ClusterID',y='Balance')
#People in Cluster 1 require highest number of miles to be eligible for award travel
sns.barplot(data=df,x='ClusterID',y='QualMiles')
#Cluster 2 contains people who require most number of miles to qualify for top flight status 
sns.barplot(data=df,x='ClusterID',y='BonusMiles')
#Cluster 1 people have the highest number of miles earned from non-flight bonus transactions in the past 12 months
sns.barplot(data=df,x='ClusterID',y='BonusTrans')
#Cluster 1 people have the highest number of non-flight bonus transactions in the past 12 months
sns.barplot(data=df,x='ClusterID',y='FlightMiles')
#Cluster 3 people have highest number of flight miles in the past 12 months, whereas we can see that flight miles are quite low

#for cluster 1 people, hence they were made to earn more flight miles through non-flight bonus transactions, so that they fly,

#and increase the business for the airline
sns.barplot(data=df,x='ClusterID',y='FlightTrans')
sns.barplot(data=df,x='ClusterID',y='DaysSinceEnroll')
##In Cluster1, people have enrolled in the flight program for a very long time, longer than others, which is why they are being

#offered more flight miles through non-flight bonus transactions, so that they can increase the frequency of flying for customers

#who have been enrolled for a long time. This hasn't had much effect on the people though. The flying miles for Cluster1 are 

#still quite less.
#Cluster0 has less flight miles, but the points they were awarded are lesser than the amount awarded to Cluster1, and that

#could be before people in cluster0 enrolled after the people in cluster1.
##Cluster 3 is not getting much fly miles through non-flight bonus transactions because they are already fliers with high miles

##and more number of transactions than the rest.
df.columns


from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
#Hierarchial Clustering:

plt.figure(figsize=(15,10))

mergings = linkage(df_norm, method='single',metric='euclidean')

dendrogram(mergings)

plt.show()
plt.figure(figsize=(15,10))

mergings = linkage(df_norm, method='complete',metric='euclidean')

dendrogram(mergings)

plt.show()
plt.figure(figsize=(15,10))

mergings = linkage(df_norm, method='average',metric='euclidean')

dendrogram(mergings)

plt.show()
##Agglomerative Clustering:
from sklearn.cluster import AgglomerativeClustering



his_clus = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')



cluster2 = his_clus.fit_predict(df3)



df_h = df3.copy(deep=True)

df_h['label'] = cluster2

df_h['label'].value_counts()
from sklearn.cluster import AgglomerativeClustering



his_clus = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')



cluster2 = his_clus.fit_predict(df3)



df_h = df3.copy(deep=True)

df_h['label'] = cluster2

df_h['label'].value_counts()
from sklearn.cluster import AgglomerativeClustering



his_clus = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')



cluster2 = his_clus.fit_predict(df3)



df_h = df3.copy(deep=True)

df_h['label'] = cluster2

df_h['label'].value_counts()
df_km['ClusterID'].value_counts()
##WE can compare what kmeans gave and what Agglomerative Clustering gave
##NOW, Principal Component Analysis:
X_std = StandardScaler().fit_transform(df3)
cov_matrix = np.cov(X_std.T)
cov_matrix
#Step3: Eigen values and eigen vector
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

print(eig_vals)

print(eig_vecs)
eigen_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted (eig_vals,reverse=True)]

cum_var_exp = np.cumsum(var_exp)

print("Cumulative Variance Explained",cum_var_exp)
df.shape[1]
plt.figure(figsize=(12,8))

plt.bar(range(7),var_exp,alpha=0.5,align='center',label='Individual Explained Variance')

plt.step(range(7),cum_var_exp,where='mid',label='Cumulative Explained Variance')

plt.ylabel('Explained Variance Ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.tight_layout()

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principal_components = pca.fit_transform(df3)

X1 = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
X1.head()
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)

plt.ylabel('percentange of explained variance')

plt.xlabel('principal component')

plt.title('Scree Plot')

plt.show()
df3.head()
plt.figure(figsize=(12,8))

sns.heatmap(df3.corr(),annot=True)
##Few of the features have high correlation, which shows that multi-collinearity will exist--one of the examples is 

#FlightMiles and FlightTrans
#So, we can consider the dataframe X1 for now, and then build a model using the PCs:
from sklearn.cluster import KMeans

Kmean = KMeans(n_clusters=2)

Kmean.fit(X1)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,

 n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',

 random_state=None, tol=0.0001, verbose=0)
Kmean.cluster_centers_
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(X1['PC1'], X1['PC2'], s =50, c='b')

plt.scatter(6.69202776e+04, -2.56491151e+01, s=200, c='g', marker='s')

plt.scatter(-2.00200830e+04, 7.67327084e+00, s=200, c='r', marker='s')

plt.show()
Kmean.labels_
X1['KMC'] = Kmean.fit_predict(X1[['PC1','PC2']])

sns.scatterplot(x='PC1',y='PC2',hue='KMC',data=X1,palette='spring')

plt.scatter(6.69202776e+04, -2.56491151e+01, s=200, marker='s')

plt.scatter(-2.00200830e+04, 7.67327084e+00, s=200, marker='s')

plt.show()
##Let's try DBSCAN for the same:
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,min_samples=10)
db.fit(X1[['PC1','PC2']])
X1['DBC'] = db.labels_
sns.scatterplot(x='PC1',y='PC2',hue='DBC',data=X1,palette='spring')
X1['DBC'].value_counts()
##So, in this case, DBSCAN was unable to classify the data into clusters
##We have already checked the inertia(Elbow plot)--let's check the Silhouette Score
from sklearn.metrics import silhouette_samples,silhouette_score
kmeans=KMeans(n_clusters=2)
X=df3
model = kmeans.fit(X=df3)
y=model.labels_
silhouette_score(X,y)
score = []

for n_clusters in range(2,10):

    kmeans = KMeans(n_clusters=n_clusters)

    kmeans.fit(X)

    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_

    score.append(silhouette_score(X, labels, metric='euclidean'))
plt.plot(score)
scoredata = pd.DataFrame(score,index=[2,3,4,5,6,7,8,9]).reset_index().rename(columns={0:'value'})
# Set the size of the plot

##Better way to plot

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)

sns.pointplot(data=scoredata,x='index',y='value')

plt.grid(True)

plt.ylabel("Silouette Score")

plt.xlabel("k")

plt.title("Silouette for K-means")