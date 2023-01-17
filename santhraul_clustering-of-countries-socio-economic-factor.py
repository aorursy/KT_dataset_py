import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

import plotly
import plotly.express as px

import pycountry

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

import warnings
warnings.filterwarnings('ignore')

# reading data
country_df = pd.read_csv('../input/country-data/Country-data.csv')
country_df.head()
# getting dataframe insights
print('shape of the data set (rows, columns) :', country_df.shape)
# Checking for data types and additional info
country_df.info()
# checkinf for null values if any
country_df.isnull().sum()
# check for duplicate rows
print('No of duplicate rows: ',country_df.duplicated().sum())
# Converting imports, exports and health spending percentages to absolute values.
country_df['exports'] = country_df['exports']*country_df['gdpp']/100
country_df['health'] = country_df['health']*country_df['gdpp']/100
country_df['imports'] = country_df['imports']*country_df['gdpp']/100
country_df.head()
# describe dataset and Checking percentiles at 25%,50%,75%,90%,95% and 99%
country_df.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Box plot
var = country_df.select_dtypes(exclude='object').columns
col = 3
row = len(var)/col+1

plt.figure(figsize=(12,12))
for i in enumerate(var):
    plt.subplot(row,col,i[0]+1)
    sns.boxplot(country_df[i[1]],color="orange")
    plt.tight_layout(pad = 2)
plt.show()

fig, axs = plt.subplots(3,3,figsize = (12,12))
# poor top 10 countries represented as `top10`

# Child Mortality Rate 
top10_child_mort = country_df[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
plt1 = sns.barplot(x='country', y='child_mort', data= top10_child_mort, ax = axs[0,0])
plt1.set(xlabel = '', ylabel= 'Child Mortality Rate')

# Exports
top10_exports = country_df[['country','exports']].sort_values('exports', ascending = True).head(10)
plt2 = sns.barplot(x='country', y='exports', data= top10_exports, ax = axs[2,1])
plt2.set(xlabel = '', ylabel= 'Exports')

# Health 
top10_health = country_df[['country','health']].sort_values('health', ascending = True).head(10)
plt3 = sns.barplot(x='country', y='health', data= top10_health, ax = axs[1,0])
plt3.set(xlabel = '', ylabel= 'Health')

# Imports
top10_imports = country_df[['country','imports']].sort_values('imports', ascending = True).head(10)
plt4 = sns.barplot(x='country', y='imports', data= top10_imports, ax = axs[2,2])
plt4.set(xlabel = '', ylabel= 'Imports')

# Per capita Income 
top10_income = country_df[['country','income']].sort_values('income', ascending = True).head(10)
plt5 = sns.barplot(x='country', y='income', data= top10_income, ax = axs[1,2])
plt5.set(xlabel = '', ylabel= 'Per capita Income')

# Inflation
top10_inflation = country_df[['country','inflation']].sort_values('inflation', ascending = False).head(10)
plt6 = sns.barplot(x='country', y='inflation', data= top10_inflation, ax = axs[2,0])
plt6.set(xlabel = '', ylabel= 'Inflation')

# Fertility Rate
top10_total_fer = country_df[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)
plt7 = sns.barplot(x='country', y='total_fer', data= top10_total_fer, ax = axs[0,1])
plt7.set(xlabel = '', ylabel= 'Fertility Rate')

# Life Expectancy
top10_life_expec = country_df[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)
plt8 = sns.barplot(x='country', y='life_expec', data= top10_life_expec, ax = axs[0,2])
plt8.set(xlabel = '', ylabel= 'Life Expectancy')

# The GDP per capita 
top10_gdpp = country_df[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)
plt9 = sns.barplot(x='country', y='gdpp', data= top10_gdpp, ax = axs[1,1])
plt9.set(xlabel = '', ylabel= 'GDP per capita')


for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.tight_layout()
# plt.savefig('eda')
plt.show()
    

# distrbution plot
var = country_df.select_dtypes(exclude='object').columns
col = 3
row = len(var)/col+1

plt.figure(figsize=(12,12))
for i in enumerate(var):
    plt.subplot(row,col,i[0]+1)
    sns.distplot(country_df[i[1]])
    plt.tight_layout(pad = 2)
plt.show()
# pairplot for continuous data type
sns.pairplot(country_df.select_dtypes(['int64','float64']), diag_kind='kde', corner=True)
plt.show()
# look at the correlation between continous varibales using heat map
plt.figure(figsize=(10,6))
sns.heatmap(country_df.corr(), annot=True, cmap='RdYlGn')
plt.show()
# Getting country code for geographical visualisation
country_geoplot = country_df.copy()
mapping = {country.name: country.alpha_3 for country in pycountry.countries}
country_geoplot['country_code'] = country_geoplot['country'].map(lambda x: mapping.get(x))
country_geoplot.head()
#gdpp
fig = px.choropleth(country_geoplot, locations="country_code",
                    color="gdpp",      
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()
#Child Moratlity
fig = px.choropleth(country_geoplot, locations="country_code",
                    color="child_mort",     
                    hover_name="country", 
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()
#Child Moratlity
fig = px.choropleth(country_geoplot, locations="country_code",
                    color="income",     
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()
# selectig numerical columns and droppig country
df = country_df.drop('country', axis=1)
# list cols for upper caping and get insigts of data
cols = ['exports', 'health', 'imports', 'total_fer','gdpp']
df[cols].describe(percentiles= [0.01,0.25,0.5,0.75,0.99])

# upper caping to 0.99 percentile
cap = 0.99
for col in cols:
    HL = round(df[col].quantile(cap),2)
    df[col] = df[col].apply(lambda x: HL if x>HL else x)
# Check values after outlier treatment
df[cols].describe(percentiles= [0.01,0.25,0.5,0.75,0.99])
# check outliers after capping
df[cols].plot.box(subplots = True, figsize = (18,6), fontsize = 12)
plt.tight_layout(pad=3)
plt.show()
# Creating scaler object
scaler = MinMaxScaler()

# fit transform
df_scaled = scaler.fit_transform(df)
df_scaled
# function hopkin statustics

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
df_scaled = pd.DataFrame(df_scaled, columns = df.columns)

# Evaluate Hopkins Statistics
print('Hopkins statistics is: ', round(hopkins(df_scaled),2))
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


plt.title('Silhoutte Score')
plt.plot(num_clusters,pd.DataFrame(ss)[0])
plt.show()
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
#chosing no. of clusters as 3 and refitting kmeans model
kmeans = KMeans(n_clusters = 3, max_iter=50,random_state = 50)
kmeans.fit(df_scaled)
#adding produced labels dataframe
country_df['KMean_ClusterID']= pd.Series(kmeans.labels_)
country_df.head()
# visualising clusters
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.scatterplot(x = 'gdpp', y = 'child_mort', data= country_df, hue = 'KMean_ClusterID', palette="bright")
plt.subplot(1,3,2)
sns.scatterplot(x = 'income', y = 'child_mort', data= country_df, hue = 'KMean_ClusterID', palette="bright")
plt.subplot(1,3,3)
sns.scatterplot(x = 'income', y = 'gdpp', data= country_df, hue = 'KMean_ClusterID', palette="bright")
plt.tight_layout()

plt.show()
# visualising clusters barplot on : `gdpp, child_mort and income` 
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.barplot(x = 'KMean_ClusterID', y = 'gdpp', data= country_df)
plt.title('GDP Percapita')

plt.subplot(1,3,2)
sns.barplot(x = 'KMean_ClusterID', y = 'child_mort', data= country_df)
plt.title('Child Mortality Rate')

plt.subplot(1,3,3)
sns.barplot(x = 'KMean_ClusterID', y = 'income', data= country_df)
plt.title('Income Per Person')

plt.tight_layout()

plt.show()
# sort based on 'child_mort','income','gdpp' in respective order
KMean_cluster_Undeveloped = country_df[country_df['KMean_ClusterID']== 2]
K_top5 = KMean_cluster_Undeveloped.sort_values(by = ['gdpp','income','child_mort'],
                                                     ascending=[True, True, False]).head(5)

print( 'Top 5 undeveloped countries based on KMean cluster are:' , K_top5['country'].values )
# Single Linkage
mergings = linkage(df_scaled, method="single", metric='euclidean')
plt.figure(figsize=(12,6))
dendrogram(mergings)
plt.title('Single Linkage')
plt.show()
# complete linkage
mergings = linkage(df_scaled, method="complete", metric='euclidean')
plt.figure(figsize=(12,6))
dendrogram(mergings)
plt.title('Complete Linkage')
plt.show()
# labeling 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels
# assign cluster labels
country_df['H_ClusterID'] = cluster_labels
country_df.head()
# visualising clusters
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.scatterplot(x = 'gdpp', y = 'child_mort', data= country_df, hue = 'H_ClusterID', palette="bright" )

plt.subplot(1,3,2)
sns.scatterplot(x = 'income', y = 'child_mort', data= country_df, hue = 'H_ClusterID', palette="bright" )

plt.subplot(1,3,3)
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
sns.scatterplot(x = 'income', y = 'gdpp', data= country_df, hue = 'H_ClusterID',palette="bright")

plt.tight_layout()
plt.show()
# visualising clusters
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.barplot(x = 'H_ClusterID', y = 'gdpp', data= country_df)
plt.title('GDP Percapita')

plt.subplot(1,3,2)
sns.barplot(x = 'H_ClusterID', y = 'child_mort', data= country_df)
plt.title('Child Mortality Rate')

plt.subplot(1,3,3)
sns.barplot(x = 'H_ClusterID', y = 'income', data= country_df)
plt.title('Income Per Person')

plt.tight_layout()

plt.show()
# sort based on 'child_mort','income','gdpp' in respective order
H_cluster_Undeveloped = country_df[country_df['H_ClusterID']== 0]
H_top5 = H_cluster_Undeveloped.sort_values(by = ['gdpp','income','child_mort'],
                                                     ascending=[True, True, False]).head(5)

print( 'Top 5 countries dire need of aid  based on H cluster are:' , H_top5['country'].values )
# No of countries in Each Distribution

H_cluster_count = country_df.H_ClusterID.value_counts()
print('Cluster wise No of Countries (Hierarchical Clustering):')
print(H_cluster_count)

K_cluster_count = country_df.KMean_ClusterID.value_counts()
print('\nCluster wise No of Countries (KMean Clustering):')
print(K_cluster_count)
# Get the % of cluster distribution
H_cluster_per = country_df.H_ClusterID.value_counts(normalize = True)*100
print('Cluster wise Countries % (Hierarchical Clustering):')
print(round(H_cluster_per,2))

K_cluster_per = country_df.KMean_ClusterID.value_counts(normalize = True)*100
print('\nCluster wise Countries % (KMean Clustering):')
print(round(K_cluster_per,2))
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
# visulaise final cluster Mean
K_cluster_mean = country_df.groupby(by = 'KMean_ClusterID').mean()

print(round(K_cluster_mean[['gdpp','income','child_mort']],2))
K_cluster_mean[['gdpp','income','child_mort']].plot.bar()
plt.yscale('log')
plt.show()
# final labels
country_df['ClusterLabels'] = country_df['KMean_ClusterID'].map({0: 'Developed', 1:'Developing', 2: 'Undeveloped'})
country_df.head()                                                      
# select final data frame for profiling
KMean_cluster = country_df[['country','gdpp','child_mort','income','ClusterLabels']]
KMean_cluster.head()
# Subset data frame based on undeveloped countries
KMean_cluster_Undeveloped = KMean_cluster[KMean_cluster['ClusterLabels'] == 'Undeveloped']
KMean_cluster_Undeveloped.head()
# sort based on 'child_mort','income','gdpp' in respective order
top5 = KMean_cluster_Undeveloped.sort_values(by = ['gdpp','income','child_mort'],
                                                     ascending=[True, True, False]).head(5)
top5 = top5[['country','gdpp','income','child_mort']]
top5
# plot for final top5 countries based on child_mort, gdpp and income
plotdata = top5.set_index('country')
plotdata.plot.bar(figsize = (8,4))
plt.show()
# Visualising top 5 undeveloped countries based on 'gdpp', 'income' and 'child_mort' (plotly library is required)
fig = px.scatter(top5, x = 'income', y = 'gdpp', 
                 animation_group="country",
                 size="child_mort",
                 color = 'country',
                 hover_name="country",
                 
                 )
fig.show()
# scater plot for bottom 5 countries based on profiling varibles
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='gdpp', y='child_mort', hue='country',
                data=top5, legend='full', palette="bright", s=300, c='lightblue')
plt.subplot(1, 3, 2)
sns.scatterplot(x='gdpp', y='income', hue='country',
                data=top5, legend='full', palette="bright", s=300, c='lightblue')
plt.subplot(1, 3, 3)
sns.scatterplot(x='income', y='child_mort', hue='country',
                data=top5, legend='full', palette="bright", s=300, c='lightblue')
plt.show()

# Get the statistics of final cluster
top5.describe()