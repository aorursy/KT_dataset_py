import numpy  as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
# load the data into dataframe



country = pd.read_csv("../input/country/Country-data.csv")

country.head()
# checking no. of rows and columns in the dataset

country.shape
# exports, import and health are given as % of of gdp. lets convert them into exact nos.

country['exports'] = country['exports']*country['gdpp']/100

country['imports'] = country['imports']*country['gdpp']/100

country['health'] = country['health']*country['gdpp']/100
# checking numeric variables

country.describe()
# checking the information

country.info()



# no null values in the dataset
# lets reconfirm the null values in the dataset

country.isnull().sum()
# lets check for outliers in the data set

cols = country.drop("country",1)



plt.figure(figsize=(20,15))

for idx,col in enumerate(cols):

    plt.subplot(3, 3, idx+1)

    sns.boxplot(y=col, data =country)

    plt.title("Box Plot for "+ col)

    

plt.show()
# lets check the correlation between different features

plt.figure(figsize = (20,10))        

sns.heatmap(country.corr(),annot = True)
# Lets apply scaling to the variables

from sklearn.preprocessing import StandardScaler



data = country.drop(["country"],axis = 1)



scaler = StandardScaler()

scaled_data =  scaler.fit_transform(data)



# lets check the dataframe with the scaled values

scaled_data
#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)
# fit the model to the scaled data

pca.fit(scaled_data)
pca.components_
#Let's check the variance ratios

print("variance explained by PCs:",pca.explained_variance_ratio_)
# cumulative variance explained by PCs

print("\ncumulative variance explained by PCs:",np.cumsum(pca.explained_variance_ratio_))



# top 3 PCs are explaining 87% variation present in the dataset
#Making the screeplot - plotting the cumulative variance against the number of components

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
#Let's try and check the first three components now

colnames = list(data.columns)

pcs_df = pd.DataFrame({ 'Feature':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2]})

pcs_df
%matplotlib inline

fig = plt.figure(figsize = (8,8))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
#Using incremental PCA for efficiency - saves a lot of time on larger datasets

from sklearn.decomposition import IncrementalPCA

pca_final = IncrementalPCA(n_components=3)
# fit and transform the scaled dataset 

df_train_pca = pca_final.fit_transform(scaled_data)

df_train_pca.shape
#creating correlation matrix for the principal components

corrmat = np.corrcoef(df_train_pca.transpose())
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (20,10))

sns.heatmap(corrmat,annot = True)
# 1s -> 0s in diagonals

corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())

print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)

# we see that correlations are indeed very close to 0
# lets take the transpose of the trained dataset

pc = np.transpose(df_train_pca)

pc
#Let's create the newer matrix according to the given principal components

rownames = list(country['country'])

pcs_df2 = pd.DataFrame({'country':rownames,'PC1':pc[0],'PC2':pc[1],'PC3':pc[2]})

pcs_df2.head()
#Let's do the outlier analysis before proceeding to clustering

plt.boxplot(pcs_df2.PC1)

Q1 = pcs_df2.PC1.quantile(0.05)

Q3 = pcs_df2.PC1.quantile(0.95)

IQR = Q3 - Q1

pcs_df2 = pcs_df2[(pcs_df2.PC1 >= Q1) & (pcs_df2.PC1 <= Q3)]
# outlier treatment for PC2

plt.boxplot(pcs_df2.PC2)

Q1 = pcs_df2.PC2.quantile(0.05)

Q3 = pcs_df2.PC2.quantile(0.95)

IQR = Q3 - Q1

pcs_df2 = pcs_df2[(pcs_df2.PC2 >= Q1) & (pcs_df2.PC2 <= Q3)]
# outlier treatment for PC3

plt.boxplot(pcs_df2.PC3)

Q1 = pcs_df2.PC3.quantile(0.05)

Q3 = pcs_df2.PC3.quantile(0.95)

IQR = Q3 - Q1

dat3 = pcs_df2[(pcs_df2.PC3 >= Q1 ) & (pcs_df2.PC3 <= Q3)]
#Outlier analysis is now done.Let's check the data again.

pcs_df2.shape
#let's check the spread of the dataset

sns.scatterplot(x='PC1',y='PC2',data=pcs_df2)
def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
from numpy.random import uniform

from random import sample

from sklearn.neighbors import NearestNeighbors

from math import isnan

hopkins(pcs_df2.drop(["country"],1))
# lets scale the dataset

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()



pcs_df3 = pcs_df2

pcs_df3 = standard_scaler.fit_transform(pcs_df3.drop(['country'],axis=1))
#Let's check the silhouette score first to identify the ideal number of clusters

from sklearn.metrics import silhouette_score

sse_ = []

for k in range(2, 10):

    kmeans = KMeans(n_clusters=k).fit(pcs_df3)

    sse_.append([k, silhouette_score(pcs_df3, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);
#Let's use the elbow curve method to identify the ideal number of clusters.

ssd = []

for num_clusters in list(range(1,10)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(pcs_df3)

    ssd.append(model_clus.inertia_)



plt.plot(ssd)
# Lets make a model with k = 5 first

model_clus5 = KMeans(n_clusters = 5, max_iter=50)

model_clus5.fit(pcs_df3)
# lets concatenate the cluster ids to the PCA transformed dataset

pcs_df4=pcs_df2

pcs_df4.index = pd.RangeIndex(len(pcs_df4.index))

dat_km = pd.concat([pcs_df4, pd.Series(model_clus5.labels_)], axis=1)

dat_km.columns = ['country', 'PC1', 'PC2','PC3','ClusterID']

dat_km.head()
# add cluster ids to the original dataset

pcs_df5=pd.merge(country,dat_km,on='country')

pcs_df6=pcs_df5[['country','child_mort','exports','imports','health','income','inflation','life_expec','total_fer','gdpp','ClusterID']]

pcs_df6.head()
pcs_df6.shape
# lets check if each cluster has enough no. of data points

pcs_df6['ClusterID'].value_counts()
# lets take mean of the available features, these would be helpful for cluster analysis

clu_chi  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).child_mort.mean())

clu_exp  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).exports.mean())

clu_imp  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).imports.mean())

clu_hea  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).health.mean())

clu_inc  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).income.mean())

clu_inf  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).inflation.mean())         

clu_lif  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).life_expec.mean())

clu_tot  = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).total_fer.mean())

clu_gdpp = pd.DataFrame(pcs_df6.groupby(["ClusterID"]).gdpp.mean())



final = pd.concat([pd.Series([0,1,2,3,4]),clu_chi,clu_exp,clu_imp,clu_hea,clu_inc,clu_inf,clu_lif,clu_tot,clu_gdpp], axis=1)

final.columns = ["ClusterID", "Child_Mortality", "Exports", "Imports","Health_Spending","Income","Inflation","Life_Expectancy","Total_Fertility","GDPpcapita"]

final
# lets visualise the cluster formed

fig= plt.figure(figsize = (25,20))



for idx,col in enumerate(final.drop("ClusterID",1)):

    plt.subplot(3,3,idx+1)

    sns.barplot(x=final.ClusterID, y=final[col])

    plt.title(col,fontsize = 15,fontweight='bold')

    

plt.show()
# from the above data, we know that average gdpp for cluster 0 is minimum i.e. 1500

# lets create a dataframe with gdpp less than 1500

poor1 = country[country.gdpp<=1500]



# income is minimum average income is around 3100, lets filter the dataset

poor2 = poor1[poor1.income<=3100]



# max average child mort is around 76, lets apply another filter

poor3 = poor2[poor2.child_mort>=76]

print("Final list of poor countries\n:",poor3)
# single linkage procedure.

mergings = linkage(pcs_df3, method = "single", metric='euclidean')

dendrogram(mergings)

plt.show()
#Let's try complete linkage method

mergings = linkage(pcs_df3, method = "complete", metric='euclidean')

dendrogram(mergings)

plt.show()
# we see good clustering here, lets cut the clusters

clusterCut = pd.Series(cut_tree(mergings, n_clusters = 5).reshape(-1,))

pcs_df2_hc = pd.concat([pcs_df2, clusterCut], axis=1)

pcs_df2_hc.columns = ['country', 'PC1', 'PC2','PC3','ClusterID']
pcs_df2_hc.head()
pcs_df7=pd.merge(country,pcs_df2_hc,on='country')

pcs_df8=pcs_df7[['country','child_mort','exports','imports','health','income','inflation','life_expec','total_fer','gdpp','ClusterID']]

pcs_df8.head()
# lets check if the clusters formed have good no. of countries 

pcs_df8['ClusterID'].value_counts()
#Cluster 4 doesn't have enough amount of clusters. Let's check other clusters

pcs_df8[pcs_df8['ClusterID']==2]
# cluster 3

pcs_df8[pcs_df8['ClusterID']==3]
# cluster 0

pcs_df8[pcs_df8['ClusterID']==0]
#looks like clusters are not formed correctly, lets visualize the clusters, we would stick with the clusters we formed earlier