import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime as dt



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
# Suppress warnings

import warnings

warnings.filterwarnings("ignore")
df_country = pd.read_csv("../input/unsupervised-learning-on-country-data/Country-data.csv", sep=",", encoding="ISO-8859-1", header=0)
df_country.head()
df_country.info()
df_country.columns
df_country.shape
# missing values

round(100*(df_country.isnull().sum())/len(df_country), 2)
#Converting exports,imports and health spending percentages to absolute values.

df_country['exports'] = df_country['exports']*df_country['gdpp']/100

df_country['imports'] = df_country['imports']*df_country['gdpp']/100

df_country['health'] = df_country['health']*df_country['gdpp']/100
df_country.head()
df_country.corr()
plt.figure(figsize = (20,10))        

sns.heatmap(df_country.corr(),annot = True, cmap="YlGnBu")
#The final matrix would only contain the data columns. Hence let's drop the country column

data=df_country.drop(['country'],axis=1)

data.head()
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

df_c = standard_scaler.fit_transform(data)
df_c.shape
fig, axs = plt.subplots(3,3,figsize = (15,15))



# Child Mortality Rate : Death of children under 5 years of age per 1000 live births



top10_child_mort = df_country[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)

plt1 = sns.barplot(x='country', y='child_mort', data= top10_child_mort, ax = axs[0,0])

plt1.set(xlabel = '', ylabel= 'Child Mortality Rate')



# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same

top10_total_fer = df_country[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)

plt1 = sns.barplot(x='country', y='total_fer', data= top10_total_fer, ax = axs[0,1])

plt1.set(xlabel = '', ylabel= 'Fertility Rate')



# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same



bottom10_life_expec = df_country[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='life_expec', data= bottom10_life_expec, ax = axs[0,2])

plt1.set(xlabel = '', ylabel= 'Life Expectancy')



# Health :Total health spending as %age of Total GDP.



bottom10_health = df_country[['country','health']].sort_values('health', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='health', data= bottom10_health, ax = axs[1,0])

plt1.set(xlabel = '', ylabel= 'Health')



# The GDP per capita : Calculated as the Total GDP divided by the total population.



bottom10_gdpp = df_country[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='gdpp', data= bottom10_gdpp, ax = axs[1,1])

plt1.set(xlabel = '', ylabel= 'GDP per capita')



# Per capita Income : Net income per person



bottom10_income = df_country[['country','income']].sort_values('income', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='income', data= bottom10_income, ax = axs[1,2])

plt1.set(xlabel = '', ylabel= 'Per capita Income')





# Inflation: The measurement of the annual growth rate of the Total GDP



top10_inflation = df_country[['country','inflation']].sort_values('inflation', ascending = False).head(10)

plt1 = sns.barplot(x='country', y='inflation', data= top10_inflation, ax = axs[2,0])

plt1.set(xlabel = '', ylabel= 'Inflation')





# Exports: Exports of goods and services. Given as %age of the Total GDP



bottom10_exports = df_country[['country','exports']].sort_values('exports', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='exports', data= bottom10_exports, ax = axs[2,1])

plt1.set(xlabel = '', ylabel= 'Exports')





# Imports: Imports of goods and services. Given as %age of the Total GDP



bottom10_imports = df_country[['country','imports']].sort_values('imports', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='imports', data= bottom10_imports, ax = axs[2,2])

plt1.set(xlabel = '', ylabel= 'Imports')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation = 90)

    

plt.tight_layout()

plt.savefig('eda')

plt.show()
fig, axs = plt.subplots(3,3, figsize = (15,7.5))

plt1 = sns.boxplot(df_country['child_mort'], ax = axs[0,0])

plt2 = sns.boxplot(df_country['health'], ax = axs[0,1])

plt3 = sns.boxplot(df_country['life_expec'], ax = axs[0,2])

plt4 = sns.boxplot(df_country['total_fer'], ax = axs[1,0])

plt5 = sns.boxplot(df_country['income'], ax = axs[1,1])

plt6 = sns.boxplot(df_country['inflation'], ax = axs[1,2])

plt7 = sns.boxplot(df_country['gdpp'], ax = axs[2,0])

plt8 = sns.boxplot(df_country['imports'], ax = axs[2,1])

plt9 = sns.boxplot(df_country['exports'], ax = axs[2,2])





plt.tight_layout()
df_country.describe()
data_c = ['child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp']



for i in data_c:

    percentiles = data[i].quantile([0.05, 0.95]).values

    data[i][data[i] <= percentiles[0]] = percentiles[0]

    data[i][data[i] >= percentiles[1]] = percentiles[1]
data.head()
# Treating (statistical) outliers

grouped_df = df_country.groupby('country')['income'].sum()

grouped_df = grouped_df.reset_index()

grouped_df.head()
Q1 = grouped_df.income.quantile(0.05)

Q3 = grouped_df.income.quantile(0.95)

IQR = Q3 - Q1

grouped_df = grouped_df[(grouped_df.income >= Q1 - 1.5*IQR) & (grouped_df.income <= Q3 + 1.5*IQR)]
standard_scaler = StandardScaler()

df_scaled = standard_scaler.fit_transform(data)

df_scaled.shape
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
hopkins(df_country.drop('country', axis = 1))
# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50, random_state=100)

kmeans.fit(df_scaled)
kmeans.labels_
ssd = []

range_n_clusters = [2,3,4,5,6,7,8,9,10,11]



for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=50)

    kmeans.fit(df_scaled)

    

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters

# ssd

plt.plot(ssd)
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8,9,10,11]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=50)

    kmeans.fit(df_scaled)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(df_scaled, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    

    
# final model with k=3

kmeans = KMeans(n_clusters=3, max_iter=50, random_state=50)

kmeans.fit(df_scaled)
kmeans.labels_
# assign the label

data['cluster_id'] = kmeans.labels_

data.head()
#plot

sns.boxplot(x='cluster_id', y='income', data=data)
#plot

sns.boxplot(x='cluster_id', y='gdpp', data=data)
#plot

sns.boxplot(x='cluster_id', y='child_mort', data=data)
fig, axs = plt.subplots(3,3, figsize = (15,7.5))

plt1 = sns.boxplot(x='cluster_id', y = 'child_mort', data=data, ax = axs[0,0])

plt2 = sns.boxplot(x='cluster_id', y = 'health', data=data, ax = axs[0,1])

plt3 = sns.boxplot(x='cluster_id', y = 'life_expec', data=data, ax = axs[0,2])

plt4 = sns.boxplot(x='cluster_id', y = 'total_fer', data=data, ax = axs[1,0])

plt5 = sns.boxplot(x='cluster_id', y = 'income', data=data, ax = axs[1,1])

plt6 = sns.boxplot(x='cluster_id', y = 'inflation', data=data, ax = axs[1,2])

plt7 = sns.boxplot(x='cluster_id', y = 'gdpp', data=data, ax = axs[2,0])

plt8 = sns.boxplot(x='cluster_id', y = 'imports', data=data, ax = axs[2,1])

plt9 = sns.boxplot(x='cluster_id', y = 'exports', data=data, ax = axs[2,2])





plt.tight_layout()
df_country['cluster_id'] = kmeans.labels_

cluster_2_kmeans = df_country[['cluster_id','country', 'child_mort', 'gdpp', 'income' ]].loc[df_country['cluster_id'] == 2].reset_index()

cluster_2_kmeans.shape
cluster_2_kmeans.sort_values(by = ['child_mort', 'gdpp', 'income'], ascending = [False, True, True]).head(5)
df_scaled = pd.DataFrame(df_scaled)

df_scaled.head()
data.head()
# single linkage

mergings = linkage(df_scaled, method="single", metric='euclidean')

dendrogram(mergings)

plt.show()
# complete linkage

mergings = linkage(df_scaled, method="complete", metric='euclidean')

dendrogram(mergings)

plt.show()
# 3 clusters

cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )

cluster_labels
# assign cluster labels

data['cluster_labels'] = cluster_labels

data.head()
# plots

fig, axs = plt.subplots(3,3, figsize = (15,7.5))

plt1 = sns.boxplot(x='cluster_labels', y = 'child_mort', data=data, ax = axs[0,0])

plt2 = sns.boxplot(x='cluster_labels', y = 'health', data=data, ax = axs[0,1])

plt3 = sns.boxplot(x='cluster_labels', y = 'life_expec', data=data, ax = axs[0,2])

plt4 = sns.boxplot(x='cluster_labels', y = 'total_fer', data=data, ax = axs[1,0])

plt5 = sns.boxplot(x='cluster_labels', y = 'income', data=data, ax = axs[1,1])

plt6 = sns.boxplot(x='cluster_labels', y = 'inflation', data=data, ax = axs[1,2])

plt7 = sns.boxplot(x='cluster_labels', y = 'gdpp', data=data, ax = axs[2,0])

plt8 = sns.boxplot(x='cluster_labels', y = 'imports', data=data, ax = axs[2,1])

plt9 = sns.boxplot(x='cluster_labels', y = 'exports', data=data, ax = axs[2,2])





plt.tight_layout()
df_country['cluster_labels'] = cluster_labels

cluster_0_hc = df_country[['cluster_labels','country', 'child_mort', 'gdpp', 'income' ]].loc[df_country['cluster_labels'] == 0].reset_index()

cluster_0_hc.shape
cluster_0_hc.sort_values(by = ['child_mort', 'gdpp', 'income'], ascending = [False, True, True]).head(5)