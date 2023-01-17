

import pandas as pd

import numpy as np



# For Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# For scaling the data

from sklearn.preprocessing import scale



# To perform K-means clustering

from sklearn.cluster import KMeans



# To perform PCA

from sklearn.decomposition import PCA



#To perform hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
#Reaidng the Dataset 



Country_data=pd.read_csv("../input/Country-data.csv")

#Reading the first 5 rows of the dataset

Country_data.head()
Country_data['exports'] = Country_data['exports']*Country_data['gdpp']/100

Country_data['imports'] = Country_data['imports']*Country_data['gdpp']/100

Country_data['health'] = Country_data['health']*Country_data['gdpp']/100
Country_data.head()
# Checking outliers at 25%,50%,75%,90%,95% and 99%

Country_data.describe(percentiles=[.25,.5,.75,.90,.95,.99])
fig = plt.figure(figsize = (12,8))

sns.boxplot(data=Country_data)

plt.show()
print("The number of countries are : ",Country_data.shape[0])
Country_data.info()
Country_data.isnull().sum()
Country_data.isna().sum()
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (20,10))

sns.heatmap(Country_data.corr(),annot = True)

plt.show()
## First let us see if we can explain the dataset using fewer variables

from sklearn.preprocessing import StandardScaler

Country_data1=Country_data.drop('country',1) ## Droping string feature country name.

standard_scaler = StandardScaler()

Country_scaled = standard_scaler.fit_transform(Country_data1)
pca = PCA(svd_solver='randomized', random_state=42)





# fiting PCA on the dataset

pca.fit(Country_scaled)
pca.components_
pca.explained_variance_ratio_
%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
colnames = list(Country_data1.columns)

pcs_df = pd.DataFrame({ 'Feature':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2],

                      'PC4':pca.components_[3],'PC5':pca.components_[4]})

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
#Finally let's go ahead and do dimenstionality reduction using the four Principal Components

from sklearn.decomposition import IncrementalPCA

pca_final = IncrementalPCA(n_components=5)
df_pca = pca_final.fit_transform(Country_scaled)

df_pca.shape
pc = np.transpose(df_pca)
corrmat = np.corrcoef(pc)


%matplotlib inline

plt.figure(figsize = (20,10))

sns.heatmap(corrmat,annot = True)

plt.show()
pcs_df2 = pd.DataFrame({'PC1':pc[0],'PC2':pc[1],'PC3':pc[2],'PC4':pc[3],'PC5':pc[4]})
fig = plt.figure(figsize = (12,8))

sns.boxplot(data=pcs_df2)

plt.show()
pcs_df2.shape
pcs_df2.head()
#Visualising the points on the PCs.

# one of the prime advatanges of PCA is that you can visualise high dimensional data

fig = plt.figure(figsize = (12,8))

sns.scatterplot(x='PC1',y='PC2',data=pcs_df2)

plt.show()
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

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
pcs_df2.info()
hopkins(pcs_df2)
pcs_df2.shape
dat3_1 = pcs_df2
from sklearn.metrics import silhouette_score

sse_ = []

for k in range(2, 10):

    kmeans = KMeans(n_clusters=k).fit(dat3_1)

    sse_.append([k, silhouette_score(dat3_1, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])

plt.show()
ssd = []

for num_clusters in list(range(1,10)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(dat3_1)

    ssd.append(model_clus.inertia_)



plt.plot(ssd)

plt.show()
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(dat3_1)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(dat3_1, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
model_clus2 = KMeans(n_clusters = 3, max_iter=50,random_state = 50)

model_clus2.fit(dat3_1)
dat4=pcs_df2

dat4.index = pd.RangeIndex(len(dat4.index))

dat_km = pd.concat([dat4, pd.Series(model_clus2.labels_)], axis=1)

dat_km.columns = ['PC1', 'PC2','PC3','PC4','PC5','ClusterID']

dat_km
dat_km['ClusterID'].value_counts()
fig = plt.figure(figsize = (12,8))

sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',legend='full',data=dat_km)



plt.title('Categories of countries on the basis of Components')

plt.show()
dat5=pd.merge(Country_data,dat_km, left_index=True,right_index=True)

dat5.head()
dat6=dat5.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)

dat6.head()
dat6.shape
Cluster_GDPP=pd.DataFrame(dat6.groupby(["ClusterID"]).gdpp.mean())

Cluster_child_mort=pd.DataFrame(dat6.groupby(["ClusterID"]).child_mort.mean())

Cluster_exports=pd.DataFrame(dat6.groupby(["ClusterID"]).exports.mean())

Cluster_income=pd.DataFrame(dat6.groupby(["ClusterID"]).income.mean())

Cluster_health=pd.DataFrame(dat6.groupby(["ClusterID"]).health.mean())

Cluster_imports=pd.DataFrame(dat6.groupby(["ClusterID"]).imports.mean())

Cluster_inflation=pd.DataFrame(dat6.groupby(["ClusterID"]).inflation.mean())

Cluster_life_expec=pd.DataFrame(dat6.groupby(["ClusterID"]).life_expec.mean())

Cluster_total_fer=pd.DataFrame(dat6.groupby(["ClusterID"]).total_fer.mean())
df = pd.concat([Cluster_GDPP,Cluster_child_mort,Cluster_income,Cluster_exports,Cluster_health,

                Cluster_imports,Cluster_inflation,Cluster_life_expec,Cluster_total_fer], axis=1)
df.columns = ["GDPP","child_mort","income","exports","health","imports","inflation","life_expec","total_fer"]

df
fig = plt.figure(figsize = (10,6))

df.rename(index={0: 'Developed Countries'},inplace=True)

df.rename(index={1: 'Developing Countries'},inplace=True)

df.rename(index={2: 'Under-developed Countries'},inplace=True)

s=sns.barplot(x=df.index,y='GDPP',data=df)

plt.xlabel('Country Groups', fontsize=10)

plt.ylabel('GDP per Capita', fontsize=10)

plt.title('Country Groups On the basis of GDPP')

plt.show()
fig = plt.figure(figsize = (10,6))

sns.barplot(x=df.index,y='income',data=df)

plt.xlabel('Country Groups', fontsize=10)

plt.title('Country Groups On the basis of Income')

plt.show()
fig = plt.figure(figsize = (10,6))

sns.barplot(x=df.index,y='child_mort',data=df)

plt.xlabel('Country Groups', fontsize=10)

plt.title('Country Groups On the basis of Child_mort Rate')

plt.show()
#Let's use the concept of binning

fin=Country_data[Country_data['gdpp']<=1909]

fin=fin[fin['child_mort']>= 92]

fin=fin[fin['income']<= 3897.35]
fin_k=pd.merge(fin,dat_km,left_index=True,right_index=True)
fin_k=fin_k.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)
fin_k.shape
fin_k_GDPP=fin_k.nsmallest(8,'gdpp')

fin_k_GDPP
fin_k_income=fin_k.nsmallest(8,'income')

fin_k_income
fin_k_mort=fin_k.nlargest(8,'child_mort')

fin_k_mort
fig = plt.figure(figsize = (12,8))

sns.scatterplot(x='gdpp',y='income',hue='ClusterID',legend='full',data=dat6)

plt.xlabel('GDP per Capita', fontsize=10)

plt.ylabel('Income per Person', fontsize=10)

plt.title('GDP per Capita vs Income per Person')

plt.show()
fig = plt.figure(figsize = (12,8))

sns.scatterplot(x='gdpp',y='child_mort',hue='ClusterID',legend='full',data=dat6)

plt.xlabel('GDP per Capita', fontsize=10)

plt.ylabel('Child_more rate', fontsize=10)

plt.title('GDP per Capita vs Child_more rate')

plt.show()
fig = plt.figure(figsize = (12,8))

sns.boxplot(x='ClusterID',y='gdpp',data=dat6)

plt.xlabel('Country Groups', fontsize=10)

plt.ylabel('GDP per Capita', fontsize=10)

plt.title('GDP per Capita of all the Country Groups')

plt.show()
fig = plt.figure(figsize = (12,8))

sns.boxplot(x='ClusterID',y='income',data=dat6)

plt.xlabel('Country Groups', fontsize=10)

plt.ylabel('Income per person', fontsize=10)

plt.title('Income per person of all the Country Groups')

plt.show()
fig = plt.figure(figsize = (12,8))

sns.boxplot(x='ClusterID',y='child_mort',data=dat6)

plt.xlabel('Country Groups', fontsize=10)

plt.ylabel('Child_mort rate', fontsize=10)

plt.title('Child_mort rate of all the Country Groups')

plt.show()
Developed_con_K=dat6[dat6['ClusterID']==0]

Avg_Developed_con_K=dat6[dat6['ClusterID']==1]

Poor_con_K=dat6[dat6['ClusterID']==2]
fig = plt.figure(figsize = (18,6))

s=sns.barplot(x='country',y='gdpp',data=Developed_con_K)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.xlabel('Country', fontsize=10)

plt.ylabel('GDP per Capita', fontsize=10)

plt.title('GDP per Capita of all the developed Countries ')

plt.show()
fig = plt.figure(figsize = (18,6))

s=sns.barplot(x='country',y='gdpp',data=Avg_Developed_con_K)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.xlabel('Country', fontsize=10)

plt.ylabel('GDP per Capita', fontsize=10)

plt.title('GDP per Capita of all the Developing Countries ')

plt.show()
fig = plt.figure(figsize = (18,6))

s=sns.barplot(x='country',y='gdpp',data=Poor_con_K)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.xlabel('Country', fontsize=10)

plt.ylabel('GDP per Capita', fontsize=10)

plt.title('GDP per Capita of all the Under-Developed Countries ')

plt.show()

fig = plt.figure(figsize = (18,6))

s=sns.barplot(x='country',y='child_mort',data=Poor_con_K)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.xlabel('Country', fontsize=10)

plt.ylabel('Child_mort Rate', fontsize=10)

plt.title('Child_mort Rate of all the Under-Developed Countries ')

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x='country',y='gdpp',data=fin_k_GDPP)

plt.title('GDPP of Top 8 the Under-Developed Countries ')

plt.xlabel('Under-Developed Countries', fontsize=10)

plt.ylabel('GDPP', fontsize=10)

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x='country',y='child_mort',data=fin_k_mort)

plt.title('Child_mort rate of Top 8 the Under-Developed Countries ')

plt.xlabel('Under-Developed Countries', fontsize=10)

plt.ylabel('Child_mort rate', fontsize=10)

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x='country',y='income',data=fin_k_income)

plt.title('Income of Top 8 the Under-Developed Countries ')

plt.xlabel('Under-Developed Countries', fontsize=10)

plt.ylabel('Income', fontsize=10)

plt.show()
pcs_df2.shape
pcs_df3 = pd.DataFrame({'PC1':pc[0],'PC2':pc[1],'PC3':pc[2],'PC4':pc[3],'PC5':pc[4]})
dat_km.head()
mergings=linkage(pcs_df2,method='single',metric='euclidean')

dendrogram(mergings)

plt.show()
#mergings=linkage(fin,method='complete',metric='euclidean')

mergings=linkage(pcs_df2,method='complete',metric='euclidean')

dendrogram(mergings)

plt.show()
cut_tree(mergings,n_clusters=3).shape
cluser_labels=cut_tree(mergings,n_clusters=3).reshape(-1,)

cluser_labels
#assign cluster labels



dat_km['Cluster_lables']=cluser_labels

dat_km.head()
dat7=pd.merge(Country_data,dat_km, left_index=True,right_index=True)

dat7.head()
dat8=dat7.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)

dat8.shape
dat8.head()
Cluster_GDPP_H=pd.DataFrame(dat8.groupby(["Cluster_lables"]).gdpp.mean())

Cluster_child_mort_H=pd.DataFrame(dat8.groupby(["Cluster_lables"]).child_mort.mean())

Cluster_income_H=pd.DataFrame(dat8.groupby(["Cluster_lables"]).income.mean())
df_H = pd.concat([Cluster_GDPP_H,Cluster_child_mort_H,Cluster_income_H], axis=1)
df_H.columns = ["GDPP","child_mort","income"]

df_H
#Let's use the concept of binning

fin_H=Country_data[Country_data['gdpp']<=2330.000000]

fin_H=fin[fin['child_mort']>= 130.000000]

fin_H=fin[fin['income']<= 5150.000000]
fin_H=pd.merge(fin_H,dat_km,left_index=True,right_index=True)
fin_H=fin_H.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)

fin_H.shape
sns.boxplot(x='Cluster_lables',y='gdpp',data=dat8)

plt.show()
sns.boxplot(x='Cluster_lables',y='child_mort',data=dat8)

plt.show()
sns.boxplot(x='Cluster_lables',y='income',data=dat8)

plt.show()
fin_H.nsmallest(8,'gdpp')