import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing Country Data
country_data = pd.read_csv('../input/country-data/Country-data.csv')
# basics of the country_data
country_data.info()
country_data.shape
country_data.describe()
country_data.isnull().sum()
#Checking if any columns having unique value ie only 1 value
unique =country_data.nunique()
unique = unique[unique.values ==1]
unique
#checking duplicates
sum(country_data.duplicated(subset = 'country')) == 0
#Looking for spelling mistakes
print(country_data['country'].unique())
country_data['exports'] = (country_data['exports']*country_data['gdpp'])/100
country_data['health'] = (country_data['health']*country_data['gdpp'])/100
country_data['imports'] = (country_data['imports']*country_data['gdpp'])/100
country_data.head()
fig, axs = plt.subplots(3,3,figsize = (15,15))

# Child Mortality Rate : Death of children under 5 years of age per 1000 live births

top10_child_mort = country_data[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
plt1 = sns.barplot(x='country', y='child_mort', data= top10_child_mort, ax = axs[0,0])
plt1.set(xlabel = '', ylabel= 'Child Mortality Rate')

# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same
top10_total_fer = country_data[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)
plt1 = sns.barplot(x='country', y='total_fer', data= top10_total_fer, ax = axs[0,1])
plt1.set(xlabel = '', ylabel= 'Fertility Rate')

# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same

bottom10_life_expec = country_data[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)
plt1 = sns.barplot(x='country', y='life_expec', data= bottom10_life_expec, ax = axs[0,2])
plt1.set(xlabel = '', ylabel= 'Life Expectancy')

# Health :Total health spending as %age of Total GDP.

bottom10_health = country_data[['country','health']].sort_values('health', ascending = True).head(10)
plt1 = sns.barplot(x='country', y='health', data= bottom10_health, ax = axs[1,0])
plt1.set(xlabel = '', ylabel= 'Health')

# The GDP per capita : Calculated as the Total GDP divided by the total population.

bottom10_gdpp = country_data[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)
plt1 = sns.barplot(x='country', y='gdpp', data= bottom10_gdpp, ax = axs[1,1])
plt1.set(xlabel = '', ylabel= 'GDP per capita')

# Per capita Income : Net income per person

bottom10_income = country_data[['country','income']].sort_values('income', ascending = True).head(10)
plt1 = sns.barplot(x='country', y='income', data= bottom10_income, ax = axs[1,2])
plt1.set(xlabel = '', ylabel= 'Per capita Income')


# Inflation: The measurement of the annual growth rate of the Total GDP

top10_inflation = country_data[['country','inflation']].sort_values('inflation', ascending = False).head(10)
plt1 = sns.barplot(x='country', y='inflation', data= top10_inflation, ax = axs[2,0])
plt1.set(xlabel = '', ylabel= 'Inflation')


# Exports: Exports of goods and services. Given as %age of the Total GDP

bottom10_exports = country_data[['country','exports']].sort_values('exports', ascending = True).head(10)
plt1 = sns.barplot(x='country', y='exports', data= bottom10_exports, ax = axs[2,1])
plt1.set(xlabel = '', ylabel= 'Exports')


# Imports: Imports of goods and services. Given as %age of the Total GDP

bottom10_imports = country_data[['country','imports']].sort_values('imports', ascending = True).head(10)
plt1 = sns.barplot(x='country', y='imports', data= bottom10_imports, ax = axs[2,2])
plt1.set(xlabel = '', ylabel= 'Imports')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.tight_layout()
plt.savefig('eda')
plt.show()
## to check Top 10 Highest Child Mortality Rated Countries
plt.figure(figsize = (10,5))
child_mort_top10 = country_data[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
ax = sns.barplot(x='country', y='child_mort', data= child_mort_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Child Mortality Rate')
plt.xticks(rotation=90)
plt.show()
## countries having highest netincome per person 
plt.figure(figsize = (10,5))
child_income_top10 = country_data[['country','income']].sort_values('income', ascending = False).head(10)
ax = sns.barplot(x='country', y='income',data= child_income_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Highest Income')
plt.xticks(rotation=90)
plt.show()
### countries having least net income per person 
plt.figure(figsize = (10,5))
child_income_least10 = country_data[['country','income']].sort_values('income', ascending = True).head(10)
ax = sns.barplot(x='country', y='income', data= child_income_least10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Least Income')
plt.xticks(rotation=90)
plt.show()
### to check the inflation rate
plt.figure(figsize = (10,5))
child_inflation = country_data[['country','inflation']].sort_values('inflation', ascending = False).head(10)
ax = sns.barplot(x='country', y='inflation', data= child_inflation)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'inflation rate')
plt.xticks(rotation=90)
plt.show()
## to check life expectation rate
plt.figure(figsize = (10,5))
child_inflation = country_data[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)
ax = sns.barplot(x='country', y='life_expec', data= child_inflation)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'life_expec')
plt.xticks(rotation=90)
plt.show()
# Heatmap to understand the attributes dependency

# Let us draw heatmap to understand the corelation better.
plt.figure(figsize = (15,10))  
sns.heatmap(country_data.corr(),annot = True,cmap="YlGnBu")
# Pairplot of all numeric columns
sns.pairplot(country_data)
country_data.describe(percentiles=[.25,.5,.75,.90,.95,.99])
plt.figure(figsize = (20,20))

feature = country_data.columns[1::1]
for i in enumerate(feature):
    plt.subplot(6,3, i[0]+1)
    sns.distplot(country_data[i[1]])
plt.figure(figsize = (20,20))

feature = country_data.columns[1::1]
for i in enumerate(feature):
    plt.subplot(6,3, i[0]+1)
    sns.boxplot(country_data[i[1]])
# As we can see there are a number of outliers in the data.

# Keeping in mind we need to identify backward countries based on socio economic and health factors.
# We will cap the outliers to values accordingly for analysis.

percentiles = country_data['child_mort'].quantile([0.01,0.99]).values
country_data['child_mort'][country_data['child_mort'] <= percentiles[0]] = percentiles[0]
country_data['child_mort'][country_data['child_mort'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['health'].quantile([0.01,0.99]).values
country_data['health'][country_data['health'] <= percentiles[0]] = percentiles[0]
country_data['health'][country_data['health'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['life_expec'].quantile([0.01,0.99]).values
country_data['life_expec'][country_data['life_expec'] <= percentiles[0]] = percentiles[0]
country_data['life_expec'][country_data['life_expec'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['total_fer'].quantile([0.01,0.99]).values
country_data['total_fer'][country_data['total_fer'] <= percentiles[0]] = percentiles[0]
country_data['total_fer'][country_data['total_fer'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['income'].quantile([0.01,0.99]).values
country_data['income'][country_data['income'] <= percentiles[0]] = percentiles[0]
country_data['income'][country_data['income'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['inflation'].quantile([0.01,0.99]).values
country_data['inflation'][country_data['inflation'] <= percentiles[0]] = percentiles[0]
country_data['inflation'][country_data['inflation'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['gdpp'].quantile([0.01,0.99]).values
country_data['gdpp'][country_data['gdpp'] <= percentiles[0]] = percentiles[0]
country_data['gdpp'][country_data['gdpp'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['imports'].quantile([0.01,0.99]).values
country_data['imports'][country_data['imports'] <= percentiles[0]] = percentiles[0]
country_data['imports'][country_data['imports'] >= percentiles[1]] = percentiles[1]

percentiles = country_data['exports'].quantile([0.01,0.99]).values
country_data['exports'][country_data['exports'] <= percentiles[0]] = percentiles[0]
country_data['exports'][country_data['exports'] >= percentiles[1]] = percentiles[1]
fig, axs = plt.subplots(3,3, figsize = (15,7.5))

plt1 = sns.boxplot(country_data['child_mort'], ax = axs[0,0])
plt2 = sns.boxplot(country_data['health'], ax = axs[0,1])
plt3 = sns.boxplot(country_data['life_expec'], ax = axs[0,2])
plt4 = sns.boxplot(country_data['total_fer'], ax = axs[1,0])
plt5 = sns.boxplot(country_data['income'], ax = axs[1,1])
plt6 = sns.boxplot(country_data['inflation'], ax = axs[1,2])
plt7 = sns.boxplot(country_data['gdpp'], ax = axs[2,0])
plt8 = sns.boxplot(country_data['imports'], ax = axs[2,1])
plt9 = sns.boxplot(country_data['exports'], ax = axs[2,2])

plt.tight_layout()
# Check the hopkins

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
hopkins(country_data.drop('country', axis = 1))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
countrydata1 = scaler.fit_transform(country_data.drop('country',axis=1))
countrydata1 = pd.DataFrame(countrydata1, columns = country_data.columns[1::1])
countrydata1.head()
# Choose the value of K
# Silhouette score
# Elbow curve-ssd

from sklearn.metrics import silhouette_score
ss = []
for k in range(2, 11):
    kmean = KMeans(n_clusters = k).fit(countrydata1)
    ss.append([k, silhouette_score(countrydata1, kmean.labels_)])
temp = pd.DataFrame(ss)    
plt.plot(temp[0], temp[1]);
ssd = []
for k in range(2, 11):
    kmean = KMeans(n_clusters = k).fit(countrydata1)
    ssd.append([k, kmean.inertia_])
    
temp = pd.DataFrame(ssd)
plt.plot(temp[0], temp[1]);
# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(countrydata1)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(countrydata1, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
# K=3
# Final Kmean Clustering

kmean = KMeans(n_clusters = 3, random_state = 50)
kmean.fit(countrydata1)
country_kmean = country_data
label  = pd.DataFrame(kmean.labels_, columns= ['cluster_id'])
label.head()
country_kmean = pd.concat([country_kmean, label], axis =1)
country_kmean.head()
country_kmean.cluster_id.value_counts()
# scatter plot for gdpp, income and clusterId
sns.scatterplot(x = 'gdpp', y = 'income', hue = 'cluster_id', data = country_kmean, palette = 'Set1');
# scatter plot for Child_mort, income and clusterId
sns.scatterplot(x = 'child_mort', y = 'income', hue = 'cluster_id', data = country_kmean, palette = 'Set1');
# scatter plot for Child_mort, gdpp and clusterId
sns.scatterplot(x = 'child_mort', y = 'gdpp', hue = 'cluster_id', data = country_kmean, palette = 'Set1');
# Making sense out of the clusters
country_kmean.drop('country', axis = 1).groupby('cluster_id').mean().plot(kind = 'bar')
# gdpp, income and child_mort
country_kmean.drop(['country', 'exports', 'health', 'imports','inflation','life_expec','total_fer'], axis = 1).groupby('cluster_id').mean().plot(kind = 'bar')
child_mort_mean =pd.DataFrame(country_kmean.groupby(["cluster_id"]).child_mort.mean())
exports_mean= pd.DataFrame(country_kmean.groupby(["cluster_id"]).exports.mean())
health_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).health.mean())
imports_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).imports.mean())
income_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).income.mean())
inflat_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).inflation.mean())
life_expec_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).life_expec.mean())
total_fer_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).total_fer.mean())
gdpp_mean = pd.DataFrame(country_kmean.groupby(["cluster_id"]).gdpp.mean())
country_analysis = pd.concat([pd.Series([0,1,2]),child_mort_mean,exports_mean,health_mean,imports_mean,income_mean,inflat_mean,life_expec_mean,
                                 total_fer_mean,gdpp_mean], axis=1)
country_analysis
country_analysis.columns = ["cluster_id","child_mort_mean","exports_mean","health_mean","imports_mean","income_mean","inflation_mean","life_expec_mean","total_fer_mean","gdpp_mean"]
country_analysis
plt.figure(figsize = (18,18))
plt.figure(1)

# subplot 1
plt.subplot(3, 3, 1)
plt.title("child_mort_mean")
sns.barplot(country_analysis.cluster_id, country_analysis.child_mort_mean)

# subplot 2 
plt.subplot(3, 3, 2)
plt.title("income_mean")
sns.barplot(country_analysis.cluster_id, country_analysis.income_mean)

# subplot 3
#plt.figure(2)
plt.subplot(3, 3, 3)
plt.title("gdpp_mean")
sns.barplot(country_analysis.cluster_id, country_analysis.income_mean)

plt.show()
country_analysis
country_analysis.columns
## so that retrieving Poor countries which need financial aid can be identified from cluster 1
cluster_kmean_final = country_kmean[country_kmean['cluster_id']==1]
cluster_kmean_final.sort_values(['gdpp','income','child_mort','health','inflation','life_expec','total_fer','imports','exports'], 
                      ascending=[True,True,False,True,False,True,False,False,True]).head(10)
## importing necessary libraries
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
countrydata1.head()
country_hc = country_data
## single linkage
mergings = linkage(countrydata1, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()
## complete linkage 
mergings = linkage(countrydata1, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels
# assign cluster labels
country_hc['cluster_labels'] = cluster_labels
country_hc.head()
country_hc.reset_index()
sns.scatterplot(x = 'gdpp', y = 'income', hue = 'cluster_labels', data = country_hc, palette = 'Set1');
sns.scatterplot(x = 'child_mort', y = 'income', hue = 'cluster_labels', data = country_hc, palette = 'Set1');
sns.scatterplot(x = 'child_mort', y = 'gdpp', hue = 'cluster_labels', data = country_hc, palette = 'Set1');
country_hc.drop('country', axis = 1).groupby('cluster_labels').mean().plot(kind = 'bar')
# gdpp, income and child_mort
country_hc.drop(['country', 'exports', 'health', 'imports','inflation','life_expec','total_fer'], axis = 1).groupby('cluster_labels').mean().plot(kind = 'bar')
child_mort_mean =pd.DataFrame(country_hc.groupby(["cluster_labels"]).child_mort.mean())
exports_mean= pd.DataFrame(country_hc.groupby(["cluster_labels"]).exports.mean())
health_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).health.mean())
imports_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).imports.mean())
income_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).income.mean())
inflat_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).inflation.mean())
life_expec_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).life_expec.mean())
total_fer_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).total_fer.mean())
gdpp_mean = pd.DataFrame(country_hc.groupby(["cluster_labels"]).gdpp.mean())
country_analysis_hc = pd.concat([pd.Series([0,1,2]),child_mort_mean,exports_mean,health_mean,imports_mean,income_mean,inflat_mean,life_expec_mean,
                                 total_fer_mean,gdpp_mean], axis=1)
country_analysis_hc

country_analysis_hc.columns = ["cluster_id","child_mort_mean","exports_mean","health_mean","imports_mean","income_mean","inflation_mean","life_expec_mean","total_fer_mean","gdpp_mean"]
country_analysis_hc
plt.figure(figsize = (18,18))
plt.figure(1)

# subplot 1
plt.subplot(3, 3, 1)
plt.title("child_mort_mean")
sns.barplot(country_analysis_hc.cluster_id, country_analysis_hc.child_mort_mean)

# subplot 2 
plt.subplot(3, 3, 2)
plt.title("income_mean")
sns.barplot(country_analysis_hc.cluster_id, country_analysis_hc.income_mean)

# subplot 3
#plt.figure(2)
plt.subplot(3, 3, 3)
plt.title("gdpp_mean")
sns.barplot(country_analysis_hc.cluster_id, country_analysis_hc.income_mean)

plt.show()
country_hc.head()
## so that retrieving Poor countries which need financial aid can be identified from cluster 0
cluster_kmean = country_hc[country_hc['cluster_labels']==0]
cluster_kmean.sort_values(['gdpp','income','child_mort','health','inflation','life_expec','total_fer','imports','exports'], 
                      ascending=[True,True,False,True,False,True,False,False,True]).head(10)