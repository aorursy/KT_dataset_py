import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/country-socioeconomic-data/Country-data.csv')

data.head()
data['country'].value_counts()
print('Categorical columns : ',list(data.select_dtypes(include='object').columns))

print('Numeric columns : ',list(data.select_dtypes(exclude='object').columns))
num=data.select_dtypes(exclude='object')

num.head()
num.describe().T
print('No of categories in the county column are :',data['country'].nunique())
print('% observations in each category :\n',data['country'].value_counts(normalize=True)*100)
corr=num.corr() #### correlation table

corr
cov=num.cov() #### covariance table

cov
plt.figure(figsize=(15,8))

sns.heatmap(corr,annot=True)

plt.show()
num.isnull().sum() ### no null values in the data
cols=list(num.columns)

for a in cols:

    sns.distplot(num[a])

    plt.show()
### Lowest 10 countries based on child_mortality

df=data[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)

sns.barplot(x='country',y='child_mort',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'total_fer'

df=data[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)

sns.barplot(x='country',y='total_fer',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'life_expec'

df=data[['country','life_expec']].sort_values('life_expec', ascending = False).head(10)

sns.barplot(x='country',y='life_expec',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'health'

df=data[['country','health']].sort_values('health', ascending = False).head(10)

sns.barplot(x='country',y='health',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'gdpp'

df=data[['country','gdpp']].sort_values('gdpp', ascending = False).head(10)

sns.barplot(x='country',y='gdpp',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'income'

df=data[['country','income']].sort_values('income', ascending = False).head(10)

sns.barplot(x='country',y='income',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'inflation'

df=data[['country','inflation']].sort_values('inflation', ascending = False).head(10)

sns.barplot(x='country',y='inflation',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'exports'

df=data[['country','exports']].sort_values('exports', ascending = False).head(10)

sns.barplot(x='country',y='exports',data=df)

plt.xticks(rotation=90)

plt.show()
### Lowest 10 countries based on 'imports'

df=data[['country','imports']].sort_values('imports', ascending = False).head(10)

sns.barplot(x='country',y='imports',data=df)

plt.xticks(rotation=90)

plt.show()
##### Scaling the data:

from scipy.stats import zscore

scale=num.apply(zscore)

scale.head()
cols=list(scale.columns)

for a in cols:

    sns.boxplot(scale[a])

    plt.show()
cov_matrics=np.cov(scale.T)

cov_matrics


eign_values , eign_vect = np.linalg.eig(cov_matrics)

print ( "Eigen Values:\n" , eign_values)

print('\n Eigen vectors : \n',eign_vect)

eig_pairs = [(eign_values[index], eign_vect[:,index]) for index in range(len(eign_values))]

eig_pairs
total = sum( eign_values )

var_exp = [ ( i / total ) * 100 for i in sorted ( eign_values , reverse = True ) ]

cum_var_exp = np.cumsum ( var_exp )

print("Cumulative Variance Explained", cum_var_exp)
plt.bar(range(1,eign_values.size + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')

plt.step(range(1,eign_values.size + 1),cum_var_exp, where= 'mid', label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc = 'best')

plt.show()
#### KMeans without PCA:

from sklearn.cluster import KMeans



wcss = []



for k in range(1,10):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(scale)

    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))

plt.plot( range(1,10), wcss, marker = "o" )
kmeans = KMeans(n_clusters=3)

kmeans.fit(scale)
kmeans.labels_
scale['Labels']=kmeans.labels_
#####  Agglomerative clustering without PCA



### MAKING OF DENDOGRAM:

from scipy.cluster.hierarchy import linkage, dendrogram,cophenet

from scipy.spatial.distance import pdist

plt.figure(figsize=[10,10])

merg = linkage(scale, method='ward')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

df_AC = scale.drop('Labels',1).copy(deep=True)

cluster2 = hie_clus.fit_predict(df_AC)





df_AC['label'] = cluster2
df_AC
scale
### PLOTTING: WITHOUT PCA



plt.title('K-Means Classes')

sns.scatterplot(x='child_mort', y='life_expec', hue='Labels', data=scale)

plt.show()

plt.title('Hierarchical Classes')

sns.scatterplot(x='child_mort', y='life_expec', hue='label', data=df_AC)

plt.show()
###### silhouette_score without pca for AC

from sklearn.metrics import silhouette_score , cohen_kappa_score

x_pca_AC=df_AC.drop('label',1)

print('silhouette_score for AC with pca :',silhouette_score (x_pca_AC , df_AC['label'] ))
###### silhouette_score without pca for Kmeans

from sklearn.metrics import silhouette_score , cohen_kappa_score

x_pca_km=scale.drop('Labels',1)

print('silhouette_score for Kmeans without pca :',silhouette_score (x_pca_km , scale['Labels'] ))
from sklearn.decomposition import PCA
###### WITH PCA

s=scale.drop('Labels',1)

p=PCA(n_components=5)

d=p.fit_transform(s)

d=pd.DataFrame(d,columns=['PC1','PC2','PC3','PC4','PC5'])

d.shape
from sklearn.cluster import KMeans



wcss = []



for k in range(1,10):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(d)

    wcss.append(kmeans.inertia_)

    

# Visualization of k values:



plt.plot(range(1,10), wcss, color='red',marker='*')

plt.title('Graph of k values and WCSS')

plt.xlabel('k values')

plt.ylabel('wcss values')

plt.show()
#### Optimal clusters are 3

km= KMeans(n_clusters=3)

km.fit(d)

d['Labels']=km.labels_
ac=d.drop('Labels',1)
###### Agglomerative clustering with PCA



from scipy.cluster.hierarchy import linkage, dendrogram,cophenet

from scipy.spatial.distance import pdist

plt.figure(figsize=[10,10])

merg = linkage(ac, method='ward')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
##### Optimal clusters are 3



#### HIERARCHICAL CLUSTERING:

from sklearn.cluster import AgglomerativeClustering



hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

cluster2 = hie_clus.fit_predict(ac)





ac['label'] = cluster2
plt.title('K-Means Classes')

sns.scatterplot(x='PC1', y='PC2', hue='Labels', style='Labels', data=d)

plt.show()

plt.title('Hierarchical Classes')

sns.scatterplot(x='PC1', y='PC2', hue='label', style='label', data=ac)

plt.show()
###### silhouette_score with pca AC 

from sklearn.metrics import silhouette_score , cohen_kappa_score

x_pca_AC=ac.drop('label',1)

print('silhouette_score for AC with pca :',silhouette_score (x_pca_AC , ac['label'] ))
###### silhouette_score with PCA Kmeans

from sklearn.metrics import silhouette_score , cohen_kappa_score

x_pca_km=d.drop('Labels',1)

print('silhouette_score for AC with pca :',silhouette_score (x_pca_km , d['Labels'] ))
#### WIth pca kmean

d.groupby('Labels').agg({'PC1':'mean','PC2':'mean','PC3':'mean','PC4':'mean','PC5':'mean'}).T
#### Kmeans boxplot

d.groupby('Labels').agg({'PC1':'mean','PC2':'mean','PC3':'mean','PC4':'mean','PC5':'mean'}).T.plot(kind='box')
#### With PCA

d1=pd.concat([d,data['country']],axis=1)
d1.groupby('Labels').agg({'country':'count'})
##### Without PCA

scale1=pd.concat([scale,data['country']],axis=1)
scale1.groupby('Labels').agg({'country':'count'})
###### USing Kmeans clustering Labels

clust_df = d1[['country','Labels']].merge(data, on = 'country')

clust_df.head()

clust_exports = pd.DataFrame(clust_df.groupby(['Labels']).exports.mean())

clust_health = pd.DataFrame(clust_df.groupby(['Labels']).health.mean())

clust_imports = pd.DataFrame(clust_df.groupby(['Labels']).imports.mean())

clust_income = pd.DataFrame(clust_df.groupby(['Labels']).income.mean())

clust_inflation = pd.DataFrame(clust_df.groupby(['Labels']).inflation.mean())

clust_life_expec = pd.DataFrame(clust_df.groupby(['Labels']).life_expec.mean())

clust_total_fer = pd.DataFrame(clust_df.groupby(['Labels']).total_fer.mean())

clust_gdpp = pd.DataFrame(clust_df.groupby(['Labels']).gdpp.mean())

clust_child_mort=pd.DataFrame(clust_df.groupby(['Labels']).child_mort.mean())
df2 = pd.concat([pd.Series(list(range(0,5))), clust_child_mort,clust_exports, clust_health, clust_imports,

               clust_income, clust_inflation, clust_life_expec,clust_total_fer,clust_gdpp], axis=1)

df2.columns = ["Labels", "child_mort_mean", "exports_mean", "health_mean", "imports_mean", "income_mean", "inflation_mean",

               "life_expec_mean", "total_fer_mean", "gdpp_mean"]

df2


fig, axs = plt.subplots(3,3,figsize = (15,15))

sns.barplot(x=df2.Labels, y=df2.child_mort_mean, ax = axs[0,0])

sns.barplot(x=df2.Labels, y=df2.exports_mean, ax = axs[0,1])

sns.barplot(x=df2.Labels, y=df2.health_mean, ax = axs[0,2])

sns.barplot(x=df2.Labels, y=df2.imports_mean, ax = axs[1,0])

sns.barplot(x=df2.Labels, y=df2.income_mean, ax = axs[1,1])

sns.barplot(x=df2.Labels, y=df2.life_expec_mean, ax = axs[1,2])

sns.barplot(x=df2.Labels, y=df2.inflation_mean, ax = axs[2,0])

sns.barplot(x=df2.Labels, y=df2.total_fer_mean, ax = axs[2,1])

sns.barplot(x=df2.Labels, y=df2.gdpp_mean, ax = axs[2,2])

plt.tight_layout()
clust_df[clust_df.Labels == 1].country.values
import pandas as pd

Country_data = pd.read_csv("../input/country-socioeconomic-data/Country-data.csv")

data_dictionary = pd.read_csv("../input/country-socioeconomic-data/data-dictionary.csv")