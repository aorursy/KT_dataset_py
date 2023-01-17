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
#supress warning
import warnings
warnings.filterwarnings('ignore')
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#import data
country_data = pd.read_csv('/kaggle/input/country-socioeconomic-data/Country-data.csv')
country_data.head()
#Above dataset countrydata export,health,imoprt are in percentage. Findout the actual value.

country_data['exports'] = country_data['exports']*country_data['gdpp']/100
country_data['health'] = country_data['health']*country_data['gdpp']/100
country_data['imports'] =country_data['imports']*country_data['gdpp']/100

country_data['exports'].apply(lambda x:round(x,2))
country_data['health'].apply(lambda x:round(x,2))
country_data['imports'].apply(lambda x:round(x,2))

country_data.head()
country_data.info()
country_data.shape
country_data.describe()
#check whether any null value present or not.

country_data.isnull().sum()
#column

country_data.columns
#numeric analysis

feature = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
list(enumerate(feature))
plt.figure(figsize=(15,10))
feature = country_data.columns[1:10]
for i in enumerate(feature):
    plt.subplot(3,3,i[0]+1)
    sns.distplot(country_data[i[1]])
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
top10_child_mort = country_data[['country','child_mort']].sort_values('child_mort',ascending=False).head(10)
sns.barplot(x='country',y='child_mort',data=top10_child_mort)
plt.xticks(rotation=90)
plt.title('Highest child Mortality counries')

plt.subplot(1,2,2)
bottom10_child_mort = country_data[['country','child_mort']].sort_values('child_mort',ascending=True).head(10)
sns.barplot(x='country',y='child_mort',data=bottom10_child_mort)
plt.xticks(rotation=90)
plt.title('Lowest child Mortality counries')

plt.show()
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
Top10_health = country_data[['country','health']].sort_values('health', ascending=False).head(10)
sns.barplot(x='country',y='health',data=Top10_health,palette='husl')
plt.xticks(rotation=90)

plt.subplot(1,2,2)
Bottom10_health = country_data[['country','health']].sort_values('health', ascending=True).head(10)
sns.barplot(x='country',y='health',data=Bottom10_health,palette='husl')
plt.xticks(rotation=90)

plt.show()

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
Top10_income = country_data[['country','income']].sort_values('income', ascending=False).head(10)
sns.barplot(x='country',y='income',data=Top10_income,palette='YlOrBr')
plt.xticks(rotation=90)
plt.title('High Income Country')

plt.subplot(1,2,2)
Bottom10_income = country_data[['country','income']].sort_values('income', ascending=True).head(10)
sns.barplot(x='country',y='income',data=Bottom10_income,palette='gist_heat')
plt.xticks(rotation=90)
plt.title('Low Income Country')

plt.show()
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
Top10_inflation = country_data[['country','inflation']].sort_values('inflation', ascending=False).head(10)
sns.barplot(x='country',y='inflation',data=Top10_inflation)
plt.xticks(rotation=90)
plt.title('High Inflation Country')

plt.subplot(1,2,2)
Bottom10_inflation = country_data[['country','inflation']].sort_values('inflation', ascending=True).head(10)
sns.barplot(x='country',y='inflation',data=Bottom10_inflation)
plt.xticks(rotation=90)
plt.title('Low Inflation Country')

plt.show()
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
Top10_import = country_data[['country','imports']].sort_values('imports', ascending=False).head(10)
sns.barplot(x='country',y='imports',data=Top10_import,palette='mako')
plt.xticks(rotation=90)
plt.title('High Import Country')

plt.subplot(1,2,2)
Top10_export = country_data[['country','exports']].sort_values('exports', ascending=False).head(10)
sns.barplot(x='country',y='exports',data=Top10_export,palette='mako')
plt.xticks(rotation=90)
plt.title('High Export Country')

plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
Top10_gdpp = country_data[['country','gdpp']].sort_values('gdpp', ascending=False).head(10)
sns.barplot(x='country',y='gdpp',data=Top10_gdpp,palette='YlOrBr')
plt.xticks(rotation=90)
plt.title('High GDPP Country')

plt.subplot(1,2,2)
plt.boxplot('gdpp',data=country_data)

plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(country_data.corr(),annot=True)
plt.title('Correlation between all features')
plt.show()
#check outliers for each column through box plot

plt.figure(figsize=(15,15))

features = country_data.columns[1:10]
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(country_data[i[1]])
#for child mortality we will cap lower end as we required the upper end actual data.
#for child_mort
q1 = country_data['child_mort'].quantile(0.05)
country_data['child_mort'][country_data['child_mort']<=q1] = q1

#for export
q4 = country_data['exports'].quantile(0.95)
country_data['exports'][country_data['exports']>=q4] = q4

#for health
q4 = country_data['health'].quantile(0.95)
country_data['health'][country_data['health']>=q4] = q4

#for import
q4 = country_data['imports'].quantile(0.95)
country_data['imports'][country_data['imports']>=q4] = q4

#for income
q4 = country_data['income'].quantile(0.95)
country_data['income'][country_data['income']>=q4] = q4

#for inflation
q4 = country_data['inflation'].quantile(0.95)
country_data['inflation'][country_data['inflation']>=q4] =q4

#for life expec
q1 = country_data['life_expec'].quantile(0.05)
country_data['life_expec'][country_data['life_expec']<=q1] = q1

#for total_fer
q4 = country_data['total_fer'].quantile(0.95)
country_data['total_fer'][country_data['total_fer']>=q4] = q4

#for gdpp
q4 = country_data['gdpp'].quantile(0.95)
country_data['gdpp'][country_data['gdpp']>=q4] = q4

#check whtehr outlier treatment has been done or not

country_data.describe()
#1st findout hopkins score for better result

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
#findout hopkins score

hopkins(country_data.drop('country',axis=1))
#for scaling drop country column

country_data1 = country_data.drop('country',axis=1)
country_data1.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
country_data1 = scaler.fit_transform(country_data1)
#convert series data to dataframe

country_data1 = pd.DataFrame(country_data1)
country_data1.columns = country_data.columns[1:10]
country_data1.head()
#silhoutte score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
ss = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k).fit(country_data1)
    ss.append([k,silhouette_score(country_data1,kmeans.labels_)])
temp = pd.DataFrame(ss)
plt.plot(temp[0],temp[1])
#elbow curve

ssd = []
for k in range(2,11):
    kmean = KMeans(n_clusters=k).fit(country_data1)
    ssd.append([k,kmean.inertia_])
temp = pd.DataFrame(ssd)
plt.plot(temp[0],temp[1])
range_n_cluster = [2,3,4,5,6,7,8]

for num_clusters in range_n_cluster:
    
    #initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(country_data1)
    
    cluster_labels = kmeans.labels_
    
    #silhouette score
    silhouette_avg = silhouette_score(country_data1,cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters,silhouette_avg))
#k=3
kmean = KMeans(n_clusters = 3,random_state=101)
kmean.fit(country_data1)
label = pd.DataFrame(kmean.labels_, columns = ['labels'])
country_data.kmean = country_data.copy()
country_data.kmean = pd.concat([country_data.kmean,label],axis=1)
country_data.kmean.head()
#how many data points we have each cluster?
country_data.kmean.labels.value_counts()
#plot

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
sns.scatterplot(x='child_mort', y='income', hue='labels', data=country_data.kmean, palette='Set1')
plt.title('Child_mort vs Income')

plt.subplot(2,2,2)
sns.scatterplot(x='child_mort', y='gdpp', hue='labels', data=country_data.kmean, palette='Set1')
plt.title('Child_mort vs GDPP')

plt.subplot(2,2,3)
sns.scatterplot(x='income', y='gdpp', hue='labels', data=country_data.kmean, palette='Set1')
plt.title('Income vs GDPP')

plt.show()
country_data.kmean.columns
#cluster profiling

country_data.kmean.drop(['country', 'exports', 'health', 'imports','inflation', 'life_expec', 'total_fer'],axis=1).groupby('labels').mean()
#plot by barplot

plt.figure(figsize=(15,10))

country_data.kmean.drop(['country','exports','health','imports','inflation','life_expec','total_fer'],axis=1).groupby('labels').mean().plot(kind='bar')
plt.show()
#plot boxplot

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.boxplot(x='labels',y='child_mort',data=country_data.kmean)

plt.subplot(2,2,2)
sns.boxplot(x='labels',y='income',data=country_data.kmean)

plt.subplot(2,2,3)
sns.boxplot(x='labels',y='gdpp',data=country_data.kmean)

plt.show()
#filter the country name

country_data.kmean[country_data.kmean['labels']==2]['country']
country_data.kmean[country_data.kmean['labels']==2].sort_values(by=['child_mort','income','gdpp'],ascending=[False,True,True]).head(5)
#import libraries
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
country_data1.head()
country_data.hirar = country_data.copy()
country_data.hirar.head()
merging = linkage(country_data1,method='single',metric='euclidean')
dendrogram(merging)
plt.show()
merging = linkage(country_data1,method='complete',metric='euclidean')
dendrogram(merging)
plt.show()
#cuttree=4

cluster_labels = cut_tree(merging, n_clusters=4).reshape(-1,)
cluster_labels
#assign cluster labels

country_data.hirar['Cluster_label'] = cluster_labels
country_data.hirar.head()
country_data.hirar.Cluster_label.value_counts()
#cluster profiling
country_data.hirar.drop(['country','exports','health','imports','inflation','life_expec','total_fer'],axis=1).groupby('Cluster_label').mean()

#Bar plot

plt.figure(figsize=(20,15))
country_data.hirar.drop(['country','exports','health','imports','inflation','life_expec','total_fer'],axis=1).groupby('Cluster_label').mean().plot(kind='bar')
plt.show()
#box plot
plt.figure(figsize=(30,20))

plt.subplot(2,2,1)
sns.boxplot(x='Cluster_label',y='child_mort',data=country_data.hirar)

plt.subplot(2,2,2)
sns.boxplot(x='Cluster_label', y='income',data=country_data.hirar)

plt.subplot(2,2,3)
sns.boxplot(x='Cluster_label',y='gdpp',data=country_data.hirar)

plt.show()
#filter country name.

country_data.hirar[country_data.hirar['Cluster_label']==0]['country']
#top-5 countries having high child_mort, low income & gdpp

country_data.hirar[country_data.hirar['Cluster_label']==0].sort_values(by=['child_mort','income','gdpp'],ascending=[False,True,True]).head(5)