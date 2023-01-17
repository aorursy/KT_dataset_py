# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os

print(os.listdir("../input"))

#loading additional dependencies 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt

#seed

import random

random.seed(30) #seed for reproducibility

#ml 

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.decomposition import PCA

#viz

import matplotlib.pyplot as plt #plots and graphs

import seaborn as sns #additional functionality and visualization

plt.style.use('fivethirtyeight')#set style

%matplotlib inline





#load data and create dataframe 

train_data = pd.read_csv('../input/train_V2.csv')

#summarize information 

print("database shape:",train_data.shape)

before = train_data.shape

print("missing data?",train_data.isnull().values.any())

print("deleting missing values...")# dataframe has missing values, we will drop them because of time constraints. Usually not desirable since missing information can actually provide with important insights.

train_data = train_data.dropna()

print("missing data?",train_data.isnull().values.any())

after = train_data.shape

#print("using random sample (1% of data) to speed up computation...")

#train_data = train_data.sample(n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=None)

print("database shape:",train_data.shape)

print("Dropped rows:",before[0]-after[0])

train_data.head()
train_data.describe()
train_data["Id"].describe()
#we will drop winning placement and all features that do not represent player behaviour

cluster_data = train_data.iloc[:,3:-2]

cluster_data= cluster_data.drop(["matchType","rankPoints","maxPlace","killPlace",

                                "killPoints","matchDuration","numGroups"],axis=1)

print("Database shape: ",cluster_data.shape)

cluster_data.head()
#developing a heatmap with example from https://seaborn.pydata.org/examples/many_pairwise_correlations.html

corr = cluster_data.corr() # compute correlation matrix

f, ax = plt.subplots(figsize=(16,16)) #set size

cmap = sns.diverging_palette(220,10,as_cmap=True) #define a custom color palette

sns.heatmap(corr,annot=False,cmap=cmap,square=True,linewidths=0.5) #draw graph

plt.show()
#top n correlations

n=10

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print(get_top_abs_correlations(corr,n))
#create footDistance feature and drop highly correlated features

cluster_data['footDistance'] = cluster_data['walkDistance'] + cluster_data['swimDistance']

cluster_data= cluster_data.drop(["kills","killStreaks","DBNOs","walkDistance","swimDistance"],axis=1)



#developing a heatmap with example from https://seaborn.pydata.org/examples/many_pairwise_correlations.html

corr = cluster_data.corr() # compute correlation matrix

f, ax = plt.subplots(figsize=(16,16)) #set size

cmap = sns.diverging_palette(220,10,as_cmap=True) #define a custom color palette

sns.heatmap(corr,annot=False,cmap=cmap,square=True,linewidths=0.5) #draw graph

plt.show()

#we also standardize the data, clustering algorithms are sensitive to scale for measuring distance

standardized = preprocessing.scale(cluster_data)

#building the df again

df_labels = cluster_data.iloc[:0].columns

st_cluster_data = pd.DataFrame(standardized, columns=df_labels)

st_cluster_data.head()
#plot distribution

plt.rcParams['figure.figsize'] = (16, 9)

plot_1 = sns.distplot(st_cluster_data["damageDealt"], kde_kws={"label": "damageDealt"})

plot_2 = sns.distplot(st_cluster_data["assists"], kde_kws={"label": "assists"})

plot_3 = sns.distplot(st_cluster_data["boosts"], kde_kws={"label": "boosts"})

plt.xlabel('Player actions distribution')
#Using the elbow method to find the optimum number of clusters

wcss = []

for i in range(1,11):

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=30)

    km.fit(st_cluster_data)

    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
# Fitting K-Means to the dataset

num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters,

                init='k-means++',

                max_iter=1000,

                n_init=20,

                random_state=30)

y_kmeans = kmeans.fit_predict(st_cluster_data)
#we can change the beginning of  the cluster numbering to 1 instead of 0 (optional)

y_kmeans1=y_kmeans

y_kmeans1=y_kmeans+1

# New Dataframe called cluster

cluster = pd.DataFrame(y_kmeans1)

# Adding cluster to the Dataset1

cluster_data['cluster'] = cluster

#Mean of clusters

kmeans_mean_cluster = pd.DataFrame(round(cluster_data.groupby('cluster').mean(),4))

#trasnponse for easier visualization

kmeans_mean_cluster
for i in range(4):

    obs = cluster_data["cluster"].where(cluster_data["cluster"]==i+1).count()

    percentage = round(cluster_data["cluster"].where(cluster_data["cluster"]==i+1).count()/cluster_data["cluster"].count()*100,2)

    print("Cluster {} has".format(i+1),obs, "players, or {}%".format(percentage))
#we standardize the data to visualize it in the same scale

radar_data = preprocessing.scale(kmeans_mean_cluster)

radar_data = pd.DataFrame(radar_data)

radar_data
#https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way

labels = np.array(cluster_data.columns.values)

labels = labels[:-1]

stats = radar_data.loc[0].values



angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))



#plot the figure

fig=plt.figure()

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title("Cluster 1: Snipers")

ax.grid(True)
#https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way

labels = np.array(cluster_data.columns.values)

labels = labels[:-1]

stats = radar_data.loc[1].values





angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))



#plot the figure

fig=plt.figure()

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title("Cluster 2: Roamers")

ax.grid(True)
#https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way

labels = np.array(cluster_data.columns.values)

labels = labels[:-1]

stats = radar_data.loc[2].values



angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))



#plot the figure

fig=plt.figure()

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title("Cluster 3: Aggressive solo")

ax.grid(True)
#https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way

labels = np.array(cluster_data.columns.values)

labels = labels[:-1]

stats = radar_data.loc[3].values



angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))



#plot the figure

fig=plt.figure()

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title("Cluster 4: Vehicle team riders")

ax.grid(True)
#we use the standardized cluster data and reduce to three dimensions 

pca = PCA(n_components=2)

pca_result = pca.fit_transform(st_cluster_data)



#https://github.com/llSourcell/spike_sorting

# Plot the 1st principal component aginst the 2nd and use the 3rd for color

fig, ax = plt.subplots(figsize=(16, 9)) 

ax.scatter(pca_result[:, 0], pca_result[:, 1])

ax.set_xlabel('1st principal component', fontsize=20)

ax.set_ylabel('2nd principal component', fontsize=20)

ax.set_title('Principal Component Analysis', fontsize=23)



fig.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()
# Fitting K-Means to the dataset

num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters,

                init='k-means++',

                max_iter=1000,

                n_init=20,

                random_state=30)

y_kmeans_pca = kmeans.fit_predict(pca_result)

y_kmeans_pca=y_kmeans_pca+1
# Plot the result

plt.scatter(pca_result[:, 0], pca_result[:, 1],

           c=y_kmeans_pca, edgecolor='none', cmap=plt.get_cmap('Spectral',4))

plt.xlabel('1st principal component', fontsize=20)

plt.ylabel('2nd principal component', fontsize=20)

plt.title('Data Clusters in 2D', fontsize=23)

plt.colorbar();