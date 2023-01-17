#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
# Reading File
df= pd.read_csv("Country-data.csv")
df.head()
df.columns
# Converting exports,health,imports in actual values as its given as %age of the GDP per capita
features_std =['exports','health','imports']
for i in features_std:
    df[i]=(df[i]*df['gdpp'])/100

df.head()
# Checking shape of dataframe
df.shape
# Checking columns type in dataframe
df.info()
# checking attributes for continuous variables
df.describe()
# distribution of continuous variables
features = df.columns[1:]
plt.figure(figsize = (20,20))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.distplot(df[i[1]])
    plt.xticks(rotation=90)
plt.show() 
# Univariate analysis & Outliers rcognition for continuous variables
features = df.columns[1:]
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(df[i[1]])
    
features = ['child_mort','exports','health','imports','inflation','life_expec','total_fer','gdpp']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.scatterplot(x='income',y=df[i[1]], data=df)
    
features = ['child_mort','exports','health','imports','income','inflation','life_expec','total_fer']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.scatterplot(x='gdpp',y=df[i[1]], data=df)
features = ['exports','health','imports','income','inflation','life_expec','total_fer','gdpp']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.scatterplot(x='child_mort',y=df[i[1]], data=df)
df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), cmap='YlGnBu',annot=True)
# Clustering was not proper beacuse of outliers,hence caping/removing outliers 
# Caping child_mort lower outliers and keeping Higher as it will be helpful in clustering more child_mort means country is in need of help
q1 = df['child_mort'].quantile(0.01)
df['child_mort'][df['child_mort']<= q1] = q1

# Caping highier outlier for other variables
q3_exports = df['exports'].quantile(0.99)
df['exports'][df['exports']>= q3_exports] = q3_exports

q3_imports = df['imports'].quantile(0.99)
df['imports'][df['imports']>= q3_imports] = q3_imports

q3_health = df['health'].quantile(0.99)
df['health'][df['health']>= q3_health] = q3_health

q3_gdpp = df['gdpp'].quantile(0.99)
df['gdpp'][df['gdpp']>= q3_gdpp] = q3_gdpp

q3_life_expec = df['life_expec'].quantile(0.99)
df['life_expec'][df['life_expec']>= q3_life_expec] = q3_life_expec

q3_income = df['income'].quantile(0.99)
df['income'][df['income']>= q3_income] = q3_income

q3_inflation = df['inflation'].quantile(0.99)
df['inflation'][df['inflation']>= q3_inflation] = q3_inflation

q3_total_fer = df['total_fer'].quantile(0.99)
df['total_fer'][df['total_fer']>= q3_total_fer] = q3_total_fer
# Checking data after caping
df.describe()
# Visualising univatriate after outlier caping
features = df.columns[1:]
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(df[i[1]])
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
hopkins(df.drop('country', axis=1))
# scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1= df.drop('country', axis=1)
df1 = scaler.fit_transform(df1)
df1 = pd.DataFrame(df1)
df1.columns = df.columns[1:]
df1.head()
# Let's find out the value of K
# Silhouette Score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
ss = []
for k in range(2,11):
    kmeans = KMeans(n_clusters = k).fit(df1)
    ss.append([k, silhouette_score(df1, kmeans.labels_)])
    silhouette_avg = silhouette_score(df1, kmeans.labels_)
    print("For n_clusters={0}, the silhouette score is {1}".format(k, silhouette_avg))
# Elbow Curve
ssd = []
for k in range(2, 11):
    kmean = KMeans(n_clusters = k).fit(df1)
    ssd.append([k, kmean.inertia_])
    
plt.plot(pd.DataFrame(ssd)[0], pd.DataFrame(ssd)[1])
# K=3 taking 3 cluster
kmean = KMeans(n_clusters = 3, random_state = 50)
kmean.fit(df1)
kmean.labels_
label = pd.DataFrame(kmean.labels_, columns = ['label'])
df.kmean = df.copy()
df.kmean = pd.concat([df.kmean, label ], axis =1)
df.kmean.head()
# How many datapoints we have in each cluster
df.kmean.label.value_counts()
df.kmean.shape
# Doing Visualisation based on child_mortality as based on that will decide for help
features = ['exports','health','imports','income','inflation','life_expec','total_fer','gdpp']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.scatterplot(x='child_mort',y=df[i[1]], hue = 'label', data = df.kmean, palette = 'Set1')
features = ['exports','health','imports','income','inflation','life_expec','total_fer','gdpp','child_mort']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.barplot(x='label', y=df[i[1]], data=df.kmean)
 
# all in one to understand better
df.kmean.drop(['country'], axis = 1).groupby('label').mean().plot(figsize=(20,12),kind = 'bar')
plt.yscale('log')
plt.show()
# Cluster Profiling: Based on GDP CHILD MORT INCOME
df.kmean.drop('country',axis = 1).groupby('label').mean()
df.kmean.drop(['country','exports', 
               'health','imports','inflation',
               'life_expec','total_fer'], axis = 1).groupby('label').mean().plot(figsize=(20,12),kind = 'bar')
plt.show()
# As in above Bar graph Child portality was not visible becase of small scale, hence taking log

df.kmean.drop(['country','exports', 
               'health','imports','inflation',
               'life_expec','total_fer'], axis = 1).groupby('label').mean().plot(figsize=(20,12),kind = 'bar')
plt.yscale('log')
plt.show()
# Getting top 5 companies which are in dire need of aid
df.kmean[df.kmean['label'] ==1].sort_values(by = ['child_mort','income', 'gdpp'], ascending = [False, True,True]).head(5)
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
df1=df.drop('country',axis=1)
mergings = linkage(df1, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()
plt.figure(figsize= (18,7))
mergings = linkage(df1, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()
# By taking horizontal cut at hieght 7000 will get 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels
df.hier = df.copy()
df.hier['label'] = cluster_labels
df.hier.head()
# How many datapoints we have in each cluster
df.hier.label.value_counts()
# Checking shape of df.hier
df.hier.shape
# Doing Visualisation based on child_mortality as based on that will decide for help
features = ['exports','health','imports','income','inflation','life_expec','total_fer','gdpp']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.scatterplot(x='child_mort',y=df[i[1]], hue = 'label', data = df.hier, palette = 'Set1')
features = ['exports','health','imports','income','inflation','life_expec','total_fer','gdpp','child_mort']
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.barplot(x='label', y=df[i[1]], data=df.hier)
# all in one to understand better
df.hier.drop(['country'], axis = 1).groupby('label').mean().plot(figsize=(20,12),kind = 'bar')
plt.yscale('log')
plt.show()
df.hier.drop('country',axis = 1).groupby('label').mean()
df.hier.drop(['country','exports', 
               'health','imports','inflation',
               'life_expec','total_fer'], axis = 1).groupby('label').mean().plot(figsize=(20,12),kind = 'bar')
plt.show()
# As in above Bar graph Child portality was not visible becase of small scale, hence taking log
plt.figure(figsize=(10,30))

df.hier.drop(['country','exports', 
               'health','imports','inflation',
               'life_expec','total_fer'], axis = 1).groupby('label').mean().plot(figsize=(20,12),kind = 'bar')
plt.yscale('log')
plt.show()
df.hier[df.hier['label'] ==0].sort_values(by = ['child_mort','income', 'gdpp' ], ascending = [False, True,True]).head(5)