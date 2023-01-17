#Importing libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



#Getting data



#I owe this to SRK's work here:

#https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/simple-exploration-notebook/notebook

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
input_variables = [x for x in df.columns.values if x not in ['id','y','timestamp']]



print("Number of predicting variables: {0}".format(len(input_variables)))
df_id_vs_variable = df[['id']+input_variables]       #Removes 'y' and 'timestamp'

df_id_vs_variable = df_id_vs_variable.fillna(0)      #Replace na by 0



df_id_vs_variable.head()
def makeBinary(x):

    if abs(x) > 0.00000:

        return 1

    else:

        return 0



df_id_vs_variable = df_id_vs_variable.groupby('id').agg('sum').applymap(makeBinary)



df_id_vs_variable.head()
df_unique_set_variables = df_id_vs_variable.drop_duplicates(keep="first")



print("Number of securities: {0}".format(df_id_vs_variable.shape[0]))

print("Number of unique lines: {0}".format(df_unique_set_variables.shape[0]))
#These lines do not correspond to any "cluster"

df_no_cluster = df_id_vs_variable.loc[~df_id_vs_variable.duplicated(input_variables,keep=False)]



#These lines are duplicated so they can be "clustered"

df_cluster = df_id_vs_variable.loc[df_id_vs_variable.duplicated(input_variables,keep=False)]



df_cluster = df_cluster.groupby(input_variables).size()



array_cluster = df_cluster.values



print("Number of securities that do not belong to a cluster:{0}".format(df_no_cluster.shape[0]))

print("Number of clusters: {0}".format(len(array_cluster)))

print("Number of securities that belong to a cluster: {0}".format(sum(array_cluster)))

print("##########################")

print("   Clusters Statistics")

print("##########################")

print(df_cluster.describe())

n, bins, patches = plt.hist(array_cluster, 50, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Cluster size')

plt.ylabel('Distribution')

plt.title(r'Distribution of Cluster Size')

plt.show()
sns.clustermap(df_id_vs_variable.transpose())
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



#Range for k

kmin = 2

kmax = 15

sil_scores = []



#Compute silouhette scoeres

for k in range(kmin,kmax):

    km = KMeans(n_clusters=k, n_init=20).fit(df_id_vs_variable)

    sil_scores.append(silhouette_score(df_id_vs_variable, km.labels_))



#Plot

plt.plot(range(kmin,kmax), sil_scores)

plt.title('KMeans Results')

plt.xlabel('Number of Clusters')

plt.ylabel('Silhouette Score')

plt.show()
n_clust = 8



km = KMeans(n_clusters=n_clust, n_init=20).fit(df_id_vs_variable)

clust = km.predict(df_id_vs_variable)
#Init table of indexes

df_clust_index = {}

for i in range(0,n_clust):

    df_clust_index[i]=[]



#Fill the cluster index

for i in range(0,len(clust)):

    df_clust_index[clust[i]].append(i)



for i in range(0,n_clust):

    df_clust_index[i] = df_id_vs_variable.iloc[df_clust_index[i]].index.values

df_clust = []



for i in range(0,n_clust):

    df_clust.append(df.loc[df.id.isin(df_clust_index[i])])

for i in range(0,n_clust):

    print(df_clust[i].shape[0])
for i in range(0,n_clust):

    n, bins, patches = plt.hist(df_clust[i].y.values, 50, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('y Value')

    plt.ylabel('Occurence')

    plt.title(r'Distribution of y Value for Cluster '+str(i))

    plt.show()