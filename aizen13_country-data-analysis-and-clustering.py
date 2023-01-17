# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#For plots

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Kmeans

from sklearn.cluster import KMeans



#PCA

from sklearn.decomposition import PCA





# suppress all warnings

import warnings

warnings.filterwarnings("ignore")
#Reading data

df = pd.read_csv("../input/country-data/Country-data.csv")

#Reading data dictionary

data_dictionary = pd.read_csv("../input/country-data/data-dictionary.csv")
#Reading data dictionary to understand the columns

data_dictionary
#Looking at data

df.head()
#Information about dataset

print('Rows:   ', df.shape[0])

print('Columns:   ', df.shape[1])

print('\n Features: \n', df.columns.tolist())

print('\n Missing values: ', df.isnull().any())

print('\n Unique Values: \n', df.nunique())
#Checking for datatypes

df.info()
df['exports'] = (df['exports']*df['gdpp'])/100

df['imports'] = (df['imports']*df['gdpp'])/100

df['health'] = (df['health']*df['gdpp'])/100
#After completing transformations

df.head()
#Let us visualise data distributions
df.columns
#Data Visualization

#Histograms

columns = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']

plt.figure(1 , figsize = (15 , 10))

n = 0 

for x in columns:

    n += 1

    plt.subplot(3 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(df[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
#Checking for outliers

#Boxplots

plt.figure(1 , figsize = (15 , 10))

sns.boxplot(data = df)

plt.show()
plt.figure(1 , figsize = (15 , 10))

sns.boxplot(data = df[['child_mort', 'inflation', 'life_expec', 'total_fer']])

plt.show()
#Pairplots to visualise relationships between variables

plt.figure(1 , figsize = (15 , 10))

sns.pairplot(df)

plt.show()
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (15,10))

sns.heatmap(df.corr(),annot = True)

plt.show()
#Dropping country column first

df1 = df.drop(['country'], axis = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df1)
df_scaled.shape
#Now performing PCA

pca = PCA(svd_solver = 'auto')

pca.fit(df_scaled)
pca.components_
pca.explained_variance_ratio_
#Plotting cumulative variance explained by PCA components

plt.figure(figsize = (15,10))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
columns = df1.columns

pcs_df = pd.DataFrame({ 'Original Feature':columns,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2],'PC4':pca.components_[3],

                       'PC5':pca.components_[4]})

pcs_df
map= pd.DataFrame(pca.components_,columns = columns)

plt.figure(figsize=(15,10))

sns.heatmap(map,cmap='twilight')
#Now performing PCA with 5 components

pca = PCA(n_components = 5, svd_solver = 'auto')

pca.fit(df_scaled)
df_final = pca.transform(df_scaled)

df_final.shape
#Converting to dataframe

df_final = pd.DataFrame(df_final, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

df_final.head()
plt.figure(figsize=(15,10))

plt.scatter(df_final['PC1'],df_final['PC2'])

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')

plt.show()
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

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
print('The Hopkins statistic for this data is')

print(hopkins(df_final))
#Creating copy of dataset to use

df_a = df_final

df_b = df_final
#Arguments for kmeans

kmeans_kwargs = {

    "init": "random",

    "n_init": 10,

    "max_iter": 300,

    "random_state": 42,

    }

   



# Detemine best value of k

# A list holds the SSE values for each k

sse = []

for k in range(1, 11):

    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)

    kmeans.fit(df_a)

    sse.append(kmeans.inertia_)
#Elbow graph

plt.plot(sse)

plt.xticks(range(0, 11))

plt.xlabel("Number of Clusters")

plt.ylabel("SSE")

plt.show()
from sklearn.metrics import silhouette_score

silhouette_coefficients = []



# Notice you start at 2 clusters for silhouette coefficient



for k in range(2, 8):

    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)

    kmeans.fit(df_b)

    score = silhouette_score(df_b, kmeans.labels_)

    silhouette_coefficients.append(score)
plt.style.use("fivethirtyeight")

plt.plot(range(2,8), silhouette_coefficients)

plt.xticks(range(2, 8))

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Coefficient")

plt.show()
df_final.head()
kmeans_kwargs = {

    "init": "random",

    "n_init": 10,

    "max_iter": 300,

    "random_state": 42,

    }



model = KMeans(n_clusters=3, **kmeans_kwargs)

model.fit(df_final)
labels = pd.DataFrame(model.labels_)

labels.columns = ['ClusterID']

labels.head()
#Creating dataset with Cluster information included

df_final = pd.concat([df_final, labels], axis=1)

df_final.head()
fig = plt.figure(figsize = (15,10))

sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',legend='full',data=df_final)



plt.title('Categories of countries on the basis of Components')

plt.show()
#Creating dataset with Cluster information included

df = pd.concat([df, labels], axis=1)

df.head()
df.shape
Cluster_GDPP=pd.DataFrame(df.groupby(["ClusterID"]).gdpp.mean())

Cluster_child_mort=pd.DataFrame(df.groupby(["ClusterID"]).child_mort.mean())

Cluster_exports=pd.DataFrame(df.groupby(["ClusterID"]).exports.mean())

Cluster_income=pd.DataFrame(df.groupby(["ClusterID"]).income.mean())

Cluster_health=pd.DataFrame(df.groupby(["ClusterID"]).health.mean())

Cluster_imports=pd.DataFrame(df.groupby(["ClusterID"]).imports.mean())

Cluster_inflation=pd.DataFrame(df.groupby(["ClusterID"]).inflation.mean())

Cluster_life_expec=pd.DataFrame(df.groupby(["ClusterID"]).life_expec.mean())

Cluster_total_fer=pd.DataFrame(df.groupby(["ClusterID"]).total_fer.mean())
cluster_data = pd.concat([Cluster_GDPP,Cluster_child_mort,Cluster_income,Cluster_exports,Cluster_health,

                Cluster_imports,Cluster_inflation,Cluster_life_expec,Cluster_total_fer], axis=1)

cluster_data.columns = ["GDPP","child_mort","income","exports","health","imports","inflation","life_expec","total_fer"]

cluster_data
#Let us create a map on that note

m = {0 : 'Developed', 1 : 'Developing', 2 : 'Under-Developed'}

df_new = df

df_new['ClusterID'] = df_new['ClusterID'].map(m)
df_new.head()
#Visualizing key metrics for different categories of countries

columns = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']

plt.figure(1 , figsize = (30 , 30))

n = 0 

for x in columns:

    n += 1

    plt.subplot(3 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    s=sns.barplot(x='ClusterID',y=x,data=df_new)

    plt.xlabel('Country Groups', fontsize=10)

    plt.ylabel(x, fontsize=10)

    plt.title('Country Groups On the basis of {}'.format(x))

plt.show()
under = df_new[df_new['ClusterID'] == 'Under-Developed']

well = df_new[df_new['ClusterID'] == 'Developed']

developing = df_new[df_new['ClusterID'] == 'Developing']

under.shape
columns = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']

plt.figure(1 , figsize = (30 , 30))

n = 0 

for x in columns:

    n += 1

    plt.subplot(3 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    s = sns.barplot(x = 'country',y = x, data = under)

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.xlabel('Countries', fontsize=10)

    plt.ylabel(x, fontsize=10)

    plt.title('{} of all the Under-Developed Countries'.format(x))

plt.show()