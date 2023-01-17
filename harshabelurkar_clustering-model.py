## Importing the libraries

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler



from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.neighbors import NearestNeighbors



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree



pd.set_option('display.max_rows', 500)
#Read the data and creating the dataframe

country_df=pd.read_csv("../input/country-data/Country-data.csv")
# Check if data is loaded or not

country_df.head()
#Converting the exports columns to its true numerical values

country_df['exports']=(country_df['exports']*country_df['gdpp'])/100
#Converting the health columns to its true numerical values

country_df['health']=(country_df['health']*country_df['gdpp'])/100
#Converting the imports columns to its true numerical values

country_df['imports']=(country_df['imports']*country_df['gdpp'])/100
# inspect the first five rows of data

country_df.head()
# check the no of rows and columns

country_df.shape
#info all the entire data along with types

country_df.info()
#Summary of the numerical columns in the dataframe

country_df.describe()
country_df.columns
# Check if any null values are present in the data

country_df.isnull().sum()
plt.figure(figsize=(20,20))

feat_list=country_df.columns[1:]

for i in enumerate(feat_list):

    plt.subplot(3,3, i[0]+1)

    sns.distplot(country_df[i[1]])
num_data=country_df[['child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp']]

sns.pairplot(num_data)

plt.show()
plt.figure(figsize=(10,20))

feat_list=country_df.columns[1:]

for i in enumerate(feat_list):

    plt.subplot(3,3,i[0]+1) 

    sns.boxplot(country_df[i[1]])



plt.show()
#Handling the outliers

# calculating in arbitary way. its is usually based on business

# removing (statistical) outliers

# 1st and 95th percentile levels



# outlier treatment for child_mort

Q1 = country_df.child_mort.quantile(0.01)

Q3 = country_df.child_mort.quantile(0.95)

country_df['child_mort'][country_df['child_mort']<=Q1]=Q1

#country_df['child_mort'][country_df['child_mort']>=Q4]=Q4



# outlier treatment for exports

Q1 = country_df.exports.quantile(0.01)

Q3 = country_df.exports.quantile(0.95)

country_df['exports'][country_df['exports']<=Q1]=Q1

country_df['exports'][country_df['exports']>=Q3]=Q3



# outlier treatment for health

Q1 = country_df.health.quantile(0.01)

Q3 = country_df.health.quantile(0.95)

country_df['health'][country_df['health']<=Q1]=Q1

country_df['health'][country_df['health']>=Q3]=Q3



# outlier treatment for imports

Q1 = country_df.imports.quantile(0.01)

Q3 = country_df.imports.quantile(0.95)

country_df['imports'][country_df['imports']<=Q1]=Q1

country_df['imports'][country_df['imports']>=Q3]=Q3



# outlier treatment for income

Q1 = country_df.income.quantile(0.01)

Q3 = country_df.income.quantile(0.95)

#country_df['income'][country_df['income']<=Q1]=Q1

country_df['income'][country_df['income']>=Q3]=Q3



# outlier treatment for inflation

Q1 = country_df.inflation.quantile(0.01)

Q3 = country_df.inflation.quantile(0.95)

country_df['inflation'][country_df['inflation']<=Q1]=Q1

country_df['inflation'][country_df['inflation']>=Q3]=Q3



# outlier treatment for life_expec

Q1 = country_df.life_expec.quantile(0.01)

Q3 = country_df.life_expec.quantile(0.95)

country_df['life_expec'][country_df['life_expec']<=Q1]=Q1

country_df['life_expec'][country_df['life_expec']>=Q3]=Q3



# outlier treatment for total_fer

Q1 = country_df.total_fer.quantile(0.01)

Q3 = country_df.total_fer.quantile(0.95)

country_df['total_fer'][country_df['total_fer']<=Q1]=Q1

country_df['total_fer'][country_df['total_fer']>=Q3]=Q3



# outlier treatment for gdpp

Q1 = country_df.gdpp.quantile(0.01)

Q3 = country_df.gdpp.quantile(0.95)

#country_df['gdpp'][country_df['gdpp']<=Q1]=Q1

country_df['gdpp'][country_df['gdpp']>=Q3]=Q3
country_df.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

country_df1 = scaler.fit_transform(country_df.drop('country', axis = 1))

country_df1
country_df1 = pd.DataFrame(country_df1, columns = country_df.columns[1:])

country_df1.head()
#Calculating the Hopkins statistic

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
hopkins(country_df1)
#running the hopkins statistics for abunt 10 times and finding the avg : As hopkins stat value changes everytime we run the code. So i prefer to take the avg if them and then see the score.

hop_list=[]

for i in range(0,9):

    hop_list.append(hopkins(country_df1))

hop_list
import statistics

statistics.mean(hop_list) 
# elbow-curve/SSD

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(country_df1)

    

    ssd.append(kmeans.inertia_)
ssd
# plot the SSDs for each n_clusters

plt.xlabel("number of clusters")

plt.ylabel("SSD")

plt.plot(range_n_clusters,ssd)

plt.show()
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(country_df1)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(country_df1, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


from sklearn.metrics import silhouette_score

ss = []

for k in range(2, 11):

    kmean = KMeans(n_clusters = k).fit(country_df1)

    ss.append([k, silhouette_score(country_df1, kmean.labels_)])

temp = pd.DataFrame(ss)    

plt.plot(temp[0], temp[1])

plt.xlabel("number of clusters")

plt.ylabel("silhouette_score")
# Kmean Clustering with k=3



kmeans = KMeans(n_clusters=3,random_state=50,max_iter=50)

kmeans.fit(country_df1)
kmeans.labels_
# assign the lcluster label

country_df['cluster_id'] = kmeans.labels_

country_df.head()
#Count of number of countires under each cluster

country_df.cluster_id.value_counts()
#scatter plot between income and gdpp with respect to Cluster Ids

sns.scatterplot(x='gdpp',y='income',hue='cluster_id',data=country_df,palette='Set1')

plt.title("Scatter plot between income and gdpp with respect to Cluster Ids")

plt.show()
#scatter plot between child_mort and gdpp with respect to Cluster Ids

sns.scatterplot(x='gdpp',y='child_mort',hue='cluster_id',data=country_df,palette='Set1')

plt.title("Scatter plot between child_mort and gdpp with respect to Cluster Ids")

plt.show()
#scatter plot between income and child_mort with respect to Cluster Ids

sns.scatterplot(x='income',y='child_mort',hue='cluster_id',data=country_df,palette='Set1')

plt.title("Scatter plot between income and child_mort with respect to Cluster Ids")

plt.show()
#Box plot which shows all the three parameters cluster wise

country_df.drop(['country','exports', 'health', 'imports', 'inflation', 'life_expec', 'total_fer'], axis = 1).groupby('cluster_id').mean().plot(kind = 'bar')

plt.title("Box plot which shows all the three parameters cluster wise")

plt.show()
#Box plot which shows all the three paramters cluster wise using log scale to makechild mortality values more evident.

country_df.drop(['country','exports', 'health', 'imports', 'inflation', 'life_expec', 'total_fer'], axis = 1).groupby('cluster_id').mean().plot(kind = 'bar')

plt.title("Box plot which shows all the three parameters cluster wise")

plt.yscale("log")

plt.show()
# plot for child_mort column for different clusters to check its variation

sns.boxplot(x='cluster_id', y='child_mort', data=country_df)

plt.title("Plot for child_mort column for different clusters to check its variation")

plt.show()
# plot for income column for different clusters to check its variation

sns.boxplot(x='cluster_id', y='income', data=country_df)

plt.title("Plot for income column for different clusters to check its variation")

plt.show()
# plot for gdpp column for different clusters to check its variation

sns.boxplot(x='cluster_id', y='gdpp', data=country_df)

plt.title("Plot for gdpp column for different clusters to check its variation")

plt.show()
#Countries under Cluster 2 ie. which have less gdpp, less income and high child mortality

country_df[country_df['cluster_id'] == 2]
country_df['country'][country_df['cluster_id'] == 2]
#Countries under Cluster 0 ie. which have medium gdpp, medium income and medium child mortality

country_df[country_df['cluster_id'] == 0]
#Countries under Cluster 1 ie. which have high gdpp, high income and low child mortality

country_df[country_df['cluster_id'] == 1]
#Get the top 5 countires that are in dire need of HELP - When Gdpp, Income and Child mortality is the order of preference.)

country_df[country_df['cluster_id'] == 2].sort_values(by = ['gdpp','income','child_mort'], ascending = [True,True,False]).head(5)
#Get the top 5 countires that are in dire need of HELP - When 'income','gdpp','child_mort is the order of preference)

country_df[country_df['cluster_id'] == 2].sort_values(by = ['income','gdpp','child_mort'], ascending = [True,True,False]).head(5)
#Get the top 5 countires that are in dire need of HELP - When 'child_mort','income','gdpp' is the order of preference)

country_df[country_df['cluster_id'] == 2].sort_values(by = ['child_mort','income','gdpp'], ascending = [False,True,True]).head(5)
country_df_hierarchy = country_df.copy()

country_df_hierarchy=country_df_hierarchy.drop(['cluster_id'],axis=1)
country_df_hierarchy.head()
# single linkage - performing on scaled df

mergings = linkage(country_df1, method="single", metric='euclidean')

dendrogram(mergings)

plt.show()
# complete linkage

plt.figure(figsize=(20,10))

mergings = linkage(country_df1, method="complete", metric='euclidean')

dendrogram(mergings)

plt.show()
# Hierarical Clustering using 3 clusters

cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )

cluster_labels
# assign cluster labels

country_df_hierarchy['cluster_labels'] = cluster_labels
country_df_hierarchy.head()
#Count of number of countires under each cluster

country_df_hierarchy.cluster_labels.value_counts()
#scatter plot between income and gdpp with respect to Cluster Ids

sns.scatterplot(x='gdpp',y='income',hue='cluster_labels',data=country_df_hierarchy,palette='Set1')

plt.title("Scatter plot between income and gdpp with respect to Cluster Ids")

plt.show()
#scatter plot between child_mort and gdpp with respect to Cluster Ids

sns.scatterplot(x='gdpp',y='child_mort',hue='cluster_labels',data=country_df_hierarchy,palette='Set1')

plt.title("Scatter plot between child_mort and gdpp with respect to Cluster Ids")

plt.show()

#scatter plot between income and child_mort with respect to Cluster Ids

sns.scatterplot(x='income',y='child_mort',hue='cluster_labels',data=country_df_hierarchy,palette='Set1')

plt.title("Scatter plot between income and child_mort with respect to Cluster Ids")

plt.show()

#Box plot which shows all the three parameters cluster wise

country_df_hierarchy.drop(['country','exports', 'health', 'imports', 'inflation', 'life_expec', 'total_fer'], axis = 1).groupby('cluster_labels').mean().plot(kind = 'bar')

plt.title("Box plot which shows all the three parameters cluster wise")

plt.show()
#Box plot which shows all the three paramters cluster wise using log scale to makechild mortality values more evident.

country_df_hierarchy.drop(['country','exports', 'health', 'imports', 'inflation', 'life_expec', 'total_fer'], axis = 1).groupby('cluster_labels').mean().plot(kind = 'bar')

plt.title("Box plot which shows all the three parameters cluster wise")

plt.yscale("log")

plt.show()
# plot for child_mort column for different clusters to check its variation

sns.boxplot(x='cluster_labels', y='child_mort', data=country_df_hierarchy)

plt.title("Plot for child_mort column for different clusters to check its variation")

plt.show()
# plot for income column for different clusters to check its variation

sns.boxplot(x='cluster_labels', y='income', data=country_df_hierarchy)

plt.title("Plot for income column for different clusters to check its variation")

plt.show()
# plot for gdpp column for different clusters to check its variation

sns.boxplot(x='cluster_labels', y='gdpp', data=country_df_hierarchy)

plt.title("Plot for gdpp column for different clusters to check its variation")

plt.show()
#Countries under Cluster 0 ie. which have less gdpp, less income and high child mortality

country_df_hierarchy[country_df_hierarchy['cluster_labels'] == 0]
country_df_hierarchy.country[country_df_hierarchy['cluster_labels'] == 0]
#Countries under Cluster 1 ie. which have medium gdpp, medium income and medium child mortality

country_df_hierarchy[country_df_hierarchy['cluster_labels'] == 1]
#Countries under Cluster 2 ie. which have high gdpp, high income and low child mortality

country_df_hierarchy[country_df_hierarchy['cluster_labels'] == 2]
#Get the top 5 countires that are in dire need to HELP #Get the top 5 countires that are in dire need to HELP - When Gdpp, Income and Child mortality is the order of preference.)

country_df_hierarchy[country_df_hierarchy['cluster_labels'] == 0].sort_values(by = ['gdpp','income','child_mort'], ascending = [True,True,False]).head(5)
#Get the top 5 countires that are in dire need to HELP - When 'income','gdpp','child_mort is the order of preference)

country_df_hierarchy[country_df_hierarchy['cluster_labels'] == 0].sort_values(by = ['income','gdpp','child_mort'], ascending = [True,True,False]).head(5)
#Get the top 5 countires that are in dire need to HELP - When 'child_mort','income','gdpp' is the order of preference)

country_df_hierarchy[country_df_hierarchy['cluster_labels'] == 0].sort_values(by = ['child_mort','income','gdpp'], ascending = [False,True,True]).head(5)