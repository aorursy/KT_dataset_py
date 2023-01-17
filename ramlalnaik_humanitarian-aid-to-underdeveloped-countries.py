# Supress warnings
import warnings
warnings.filterwarnings ('ignore')
# Importing required packages
import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
%matplotlib inline
from pylab import rcParams
# reading datasets
df = pd.read_csv('../input/pca-kmeans-hierarchical-clustering/Country-data.csv')
df.head()
# Checking number of columns and rows of the dataframe using shape function
df.shape
# Checking basic information of the dataset
df.info()
# Checking the descriptive statistics
df.describe()
# Checking the columns names
df.columns
# Cheking the missing/values in the dataset
df.isnull().sum()
# Cheking the % of the missing/values in the NGO dataset
round(100*df.isnull().sum()/len(df.index),2)
# Checking the duplicates
sum(df.duplicated(subset = 'country'))==0
# Checking the duplicates and dropping the entire duplicated rows if any
df.drop_duplicates(subset = None, inplace =True)
df.shape
#Checking the values before transformation
df.head()
# Converting imports, exports and health spending percentages to absolute values.

df['imports'] = df['imports'] * df['gdpp']/100
df['exports'] = df['exports'] * df['gdpp']/100
df['health'] = df['health'] * df['gdpp']/100

df.head()
# Checking the new shape of the dataframe
df.shape
# selectig numerical columns and droppig country
# list cols for upper caping
cols = ['exports', 'health', 'imports', 'total_fer','gdpp']
df[cols].describe(percentiles= [0.01,0.25,0.5,0.75,0.99])

# describe dataset and Checking outliers at 25%,50%,75%,90%,95% and 99%
df.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Cheking the outliers - how values in each columns are distrivuted using boxplot
fig, axs = plt.subplots(3,3, figsize = (15,12))

plt1 = sns.boxplot(df['child_mort'], ax = axs[0,0], color = 'orange')
plt2 = sns.boxplot(df['exports'], ax = axs[0,1])
plt3 = sns.boxplot(df['health'], ax = axs[0,2])
plt4 = sns.boxplot(df['imports'], ax = axs[1,0])
plt5 = sns.boxplot(df['income'], ax = axs[1,1], color = 'orange')
plt6 = sns.boxplot(df['inflation'], ax = axs[1,2])
plt7 = sns.boxplot(df['life_expec'], ax = axs[2,0])
plt8 = sns.boxplot(df['total_fer'], ax = axs[2,1])
plt9 = sns.boxplot(df['gdpp'], ax = axs[2,2], color = 'orange')

plt.show()
df.columns

fig, axs = plt.subplots(3,3,figsize = (15,15))

# poor top 10 countries represented as `pt10`

# Child Mortality Rate 
pt10_child_mort = df[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
plt1 = sns.barplot(x='country', y='child_mort', data= pt10_child_mort, ax = axs[0,0])
plt1.set(xlabel = '', ylabel= 'Child Mortality Rate')

# Exports
pt10_exports = df[['country','exports']].sort_values('exports', ascending = True).head(10)
plt2 = sns.barplot(x='country', y='exports', data= pt10_exports, ax = axs[2,1])
plt2.set(xlabel = '', ylabel= 'Exports')

# Health 
pt10_health = df[['country','health']].sort_values('health', ascending = True).head(10)
plt3 = sns.barplot(x='country', y='health', data= pt10_health, ax = axs[1,0])
plt3.set(xlabel = '', ylabel= 'Health')

# Imports
pt10_imports = df[['country','imports']].sort_values('imports', ascending = True).head(10)
plt4 = sns.barplot(x='country', y='imports', data= pt10_imports, ax = axs[2,2])
plt4.set(xlabel = '', ylabel= 'Imports')

# Per capita Income 
pt10_income = df[['country','income']].sort_values('income', ascending = True).head(10)
plt5 = sns.barplot(x='country', y='income', data= pt10_income, ax = axs[1,2])
plt5.set(xlabel = '', ylabel= 'Per capita Income')

# Inflation
pt10_inflation = df[['country','inflation']].sort_values('inflation', ascending = False).head(10)
plt6 = sns.barplot(x='country', y='inflation', data= pt10_inflation, ax = axs[2,0])
plt6.set(xlabel = '', ylabel= 'Inflation')

# Fertility Rate
pt10_total_fer = df[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)
plt7 = sns.barplot(x='country', y='total_fer', data= pt10_total_fer, ax = axs[0,1])
plt7.set(xlabel = '', ylabel= 'Fertility Rate')

# Life Expectancy
pt10_life_expec = df[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)
plt8 = sns.barplot(x='country', y='life_expec', data= pt10_life_expec, ax = axs[0,2])
plt8.set(xlabel = '', ylabel= 'Life Expectancy')

# The GDP per capita 
pt10_gdpp = df[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)
plt9 = sns.barplot(x='country', y='gdpp', data= pt10_gdpp, ax = axs[1,1])
plt9.set(xlabel = '', ylabel= 'GDP per capita')


for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.tight_layout()
plt.show()
    

# Distribution Plot
plt.figure(figsize=(15, 15))
features = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
for i in enumerate(features):
    ax = plt.subplot(3, 3, i[0]+1)
    sns.distplot(df[i[1]])
    plt.xticks(rotation=20)
# pairplot for continuous data type
sns.pairplot(df.select_dtypes(['int64','float64']), diag_kind='kde', corner=True)
plt.show()
# Corrleation of the df dataset
plt.figure(figsize= (12,8))
sns.heatmap(df.corr(), annot = True, cmap = "YlGnBu")
# Selecting the numerical columns and dropping the country
df1 = df.drop('country', axis =1)
# list cols for upper caping and get insigts of data
cols = ['exports', 'health', 'imports', 'total_fer','gdpp']
df1[cols].describe(percentiles= [0.01,0.25,0.5,0.75,0.99])
# upper caping
cap = 0.99
for col in cols:
    HL = round(df1[col].quantile(cap),2)
    df1[col] = df1[col].apply(lambda x: HL if x>HL else x)
# Descriptive statistics after capping
df1[cols].describe(percentiles= [0.01,0.25,0.5,0.75,0.99])
# check outliers after capping
df1[cols].plot.box(subplots = True, figsize = (18,6), fontsize = 12)
plt.tight_layout(pad=3)
plt.show()
# dataset columns
df1.columns
# Create a scaling object
scaler = MinMaxScaler()

# fit_transform
df_scaled = scaler.fit_transform(df1)
df_scaled.shape
df_scaled
# function hopkin statustics

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
# Create dataframe of sclaled fetaures
df_scaled = pd.DataFrame(df_scaled, columns = df1.columns)

# Evaluate Hopkins Statistics
print('Hopkins statistics is: ', round(hopkins(df_scaled),2))
# CHeking the dataframe
df_scaled.head()
# Check shape
df_scaled.shape
# Creating list of clusters for no of cluster
num_clusers = list(range(1,11))
ssd = []
for clustuer in num_clusers:
    kmeans = KMeans(n_clusters=clustuer, max_iter= 50)
    kmeans.fit(df_scaled)
    ssd.append(kmeans.inertia_)
# pltng elbow method plot
plt.figure(figsize=(10,6))
plt.plot(num_clusers,ssd, marker = 'o')
plt.title('Elbow Method', fontsize = 16)
plt.xlabel('Number of clusters',fontsize=12)
plt.ylabel('Sum of Squared distance',fontsize=12)
plt.vlines(x=3, ymax=ssd[-1], ymin=ssd[0], colors="r", linestyles="-")
plt.hlines(y=ssd[2], xmax=9, xmin=1, colors="r", linestyles="--")

plt.show()
# silhouette analysis
num_clusters = list(range(2,11))
ss = []
for cluster in num_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters= cluster, max_iter=50)
    kmeans.fit(df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = round(silhouette_score(df_scaled, cluster_labels),4)
    ss.append(silhouette_avg)
    print("For n_clusters={0}, the silhouette score is {1}".format(cluster, silhouette_avg))


plt.plot(num_clusters,pd.DataFrame(ss)[0])
plt.title('Silhouette Score', fontsize = 16)
plt.show()
# K-Mean with k =3
kmeans = KMeans(n_clusters = 3, max_iter = 50, random_state= 50)
kmeans.fit(df_scaled)
kmeans.labels_
#adding produced labels dataframe
df_country = df.copy()
df_country['KMean_clusterid']= pd.Series(kmeans.labels_)
df_country.head()
# Checking the no of countries in each cluster
df_country.KMean_clusterid.value_counts()
# Scatter plot on various variables to visualize the clusters based on them

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='gdpp', y='child_mort', hue='KMean_clusterid', data=df_country, palette="bright", alpha=.4)

plt.subplot(1, 3, 2)
sns.scatterplot(x='income', y='child_mort', hue='KMean_clusterid',data=df_country, palette="bright", alpha=.4)

plt.subplot(1, 3, 3)
sns.scatterplot(x='gdpp', y='income', hue='KMean_clusterid', data=df_country, palette="bright", alpha=.4)

plt.show()
# visualising clusters
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
sns.barplot(x = 'KMean_clusterid', y = 'gdpp', data=df_country, palette="bright")
plt.title('GDP Percapita')

plt.subplot(1,3,2)
sns.barplot(x = 'KMean_clusterid', y = 'child_mort', data=df_country, palette="bright")
plt.title('Child Mortality Rate')

plt.subplot(1,3,3)
sns.barplot(x = 'KMean_clusterid', y = 'income', data=df_country, palette="bright")
plt.title('Income Per Person')

plt.tight_layout()

plt.show()
# Cheking the cluter means
df_country.groupby(['KMean_clusterid']).mean().sort_values(['child_mort','income','gdpp'],ascending = [False,True,True])
#New dataframe for group by & analysis

df_country_analysis =  df_country.groupby(['KMean_clusterid']).mean().sort_values(['child_mort','income','gdpp'],ascending = [False,True,True])
df_country_analysis
# Creating a new field for count of observations in each cluster

df_country_analysis['Observations']=df_country[['KMean_clusterid','child_mort']].groupby(['KMean_clusterid']).count()
df_country_analysis
# Creating a new field for proportion of observations in each cluster

df_country_analysis['Proportion']=round(df_country_analysis['Observations']/df_country_analysis['Observations'].sum(),2)


#Summary View
df_country_analysis[['child_mort','income','gdpp','Observations','Proportion']]
# Plot1 between income and gdpp against cluster_lables3
plt.figure (figsize = (15,10))

df_country_plot1 = df_country[['KMean_clusterid', 'gdpp', 'income']].copy()
df_country_plot1 = df_country_plot1.groupby('KMean_clusterid').mean()
df_country_plot1.plot.bar()

plt.tight_layout()
plt.show()
# Plot 2 between child_mort and cluster_labels

plt.figure (figsize = (15,10))

df_country_plot2 = df_country[['KMean_clusterid', 'child_mort']].copy()
df_country_plot2 = df_country_plot2.groupby('KMean_clusterid').mean()
df_country_plot2.plot.bar()

plt.tight_layout()
plt.show()

# sort based on 'child_mort','income','gdpp' in respective order
K_cluster_Undeveloped = df_country[df_country['KMean_clusterid']== 2]
K_top5 = K_cluster_Undeveloped.sort_values(by = ['gdpp','income','child_mort'],
                                                     ascending=[True, True, False]).head(5)

print( 'Top 5 countries dire need of aid  based on K cluster are:' , K_top5['country'].values )
# new dataset check
df_scaled.head()
#  Utilise the single linkage method for clustering this dataset 
plt.figure(figsize = (18,8))
mergings = linkage(df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()
#  Utilise the complete linkage method for clustering this dataset.

plt.figure(figsize = (18,8))
mergings = linkage(df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()
# Creating the labels
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels
# assign cluster labels
#df_country = df.copy()
df_country['H_ClusterID'] = pd.Series(cluster_labels)
df_country.head()
# Cheking the new dataframe shape
df_country.shape
# Scatter plot on various variables to visualize the clusters based on them

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='gdpp', y='child_mort', hue='H_ClusterID', data=df_country, palette="bright", alpha=.4)

plt.subplot(1, 3, 2)
sns.scatterplot(x='income', y='child_mort', hue='H_ClusterID',data=df_country, palette="bright", alpha=.4)

plt.subplot(1, 3, 3)
sns.scatterplot(x='gdpp', y='income', hue='H_ClusterID', data=df_country, palette="bright", alpha=.4)

plt.show()
# visualising clusters
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
sns.barplot(x = 'H_ClusterID', y = 'gdpp', data=df_country, palette="bright")
plt.title('GDP Percapita')

plt.subplot(1,3,2)
sns.barplot(x = 'H_ClusterID', y = 'child_mort', data=df_country, palette="bright")
plt.title('Child Mortality Rate')

plt.subplot(1,3,3)
sns.barplot(x = 'H_ClusterID', y = 'income', data=df_country, palette="bright")
plt.title('Income Per Person')

plt.tight_layout()

plt.show()
# Cheking the cluster count
df_country.H_ClusterID.value_counts()
# Checking the countries in Cluster 2 to see which are the countries in that segment.
cluster_2 = df_country[df_country['H_ClusterID']==2]
cluster_2.head()
# Checking the countries in Cluster 1 to see which are the countries in that segment.
cluster_1 = df_country[df_country['H_ClusterID']== 1]
cluster_1.head()
#New dataframe for group by & analysis
df_country_analysis = df_country.groupby(['H_ClusterID']).mean()
df_country_analysis
# Creating a new field for count of observations in each cluster
df_country_analysis['Observations'] = df_country[['H_ClusterID', 'child_mort']].groupby(['H_ClusterID']).count()
df_country_analysis
# Creating a new field for proportion of observations in each cluster
df_country_analysis['Proportion'] = round(df_country_analysis ['Observations'] / (df_country_analysis ['Observations'].sum()),2)
df_country_analysis
# Plot1 between income and gdpp against cluster_lables3
plt.figure (figsize = (15,10))

df_country_plot1 = df_country[['H_ClusterID', 'gdpp', 'income']].copy()
df_country_plot1 = df_country_plot1.groupby('H_ClusterID').mean()
df_country_plot1.plot.bar()

plt.tight_layout()
plt.show()
# Plot 2 between child_mort and cluster_labels

plt.figure (figsize = (15,10))

df_country_plot2 = df_country[['H_ClusterID', 'child_mort']].copy()
df_country_plot2 = df_country_plot2.groupby('H_ClusterID').mean()
df_country_plot2.plot.bar()

plt.tight_layout()
plt.show()

# sort based on 'child_mort','income','gdpp' in respective order
H_cluster_Undeveloped = df_country[df_country['H_ClusterID']== 0]
H_top5 = H_cluster_Undeveloped.sort_values(by = ['gdpp','income','child_mort'],
                                                     ascending=[True, True, False]).head(5)

print( 'Top 5 countries dire need of aid  based on H cluster are:' , H_top5['country'].values )
# Get the % of cluster distribution
H_cluster_per = df_country.H_ClusterID.value_counts(normalize = True)*100
print('Hierarchical Clustering Countries %:')
print(df_country.H_ClusterID.value_counts(normalize = True)*100)

K_cluster_per = df_country.KMean_clusterid.value_counts(normalize = True)*100
print('\nKMean Clustering Countries %:')
print(df_country.KMean_clusterid.value_counts(normalize = True)*100)
# barplot for cluster distribution
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
sns.barplot(x= H_cluster_per.index, y = H_cluster_per)
plt.title('Hierarchical Cluster Distribution')
plt.xlabel('ClusterID')
plt.ylabel('Percentage Distribution')

plt.subplot(1,2,2)
sns.barplot(x= K_cluster_per.index, y = K_cluster_per)
plt.title('KMean Cluster Distribution')
plt.xlabel('ClusterID')
plt.ylabel('Percentage Distribution')
plt.show()
# Final labels
df_country['ClusterLabels'] = df_country['KMean_clusterid'].map({0: 'Developed', 1:'Developing', 2: 'Undeveloped'})
df_country.head()                                                      
# select final data frame for profiling
K_cluster = df_country[['country','gdpp','child_mort','income','ClusterLabels']]
K_cluster.head()
# Subset data frame based on undeveloped countries
K_cluster_UDC = K_cluster[K_cluster['ClusterLabels'] == 'Undeveloped']
K_cluster_UDC.head()
# sort based on 'child_mort','income','gdpp' in respective order
K_top5=K_cluster_UDC.sort_values(by = ['gdpp','income', 'child_mort']).head(5).copy()

K_top5 = K_top5[['country','gdpp','income', 'child_mort']]
#Final country list
K_top5
# plot for final top5 countries based on child_nort, gdpp and income

KMean_plot = K_top5.set_index('country')
KMean_plot.plot.bar(figsize = (8,4))

plt.show()
# Bivariate Analysis of Cluster 'Under_Developed_Countries' (recommended 5)

# Scatter plot on various variables to visualize the clusters based on them

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='gdpp', y='child_mort', hue='country',
                data=K_top5, legend='full', palette="bright", s=300, c='lightblue')
plt.subplot(1, 3, 2)
sns.scatterplot(x='gdpp', y='income', hue='country',
                data=K_top5, legend='full', palette="bright", s=300, c='lightblue')
plt.subplot(1, 3, 3)
sns.scatterplot(x='income', y='child_mort', hue='country',
                data=K_top5, legend='full', palette="bright", s=300, c='lightblue')
plt.show()

# Descriptive Statistics
K_top5.describe()
#TOP COUNTRIES recommended for Financial bases on KMean Clustering analysis
print( 'Top 5 countries dire need of aid:' , K_top5['country'].values )