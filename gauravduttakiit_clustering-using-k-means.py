# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline
# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)
# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

ngo= pd.read_csv(r"/kaggle/input/help-international/Country-data.csv")

ngo.head()
word=pd.read_csv(r"/kaggle/input/help-international/data-dictionary.csv")

word.head(len(word))
ngo_dub = ngo.copy()



# Checking for duplicates and dropping the entire duplicate row if any

ngo_dub.drop_duplicates(subset=None, inplace=True)

ngo_dub.shape
ngo.shape
ngo.shape
ngo.info()
ngo.describe()
(ngo.isnull().sum() * 100 / len(ngo)).value_counts(ascending=False)
ngo.isnull().sum().value_counts(ascending=False)
(ngo.isnull().sum(axis=1) * 100 / len(ngo)).value_counts(ascending=False)
ngo.isnull().sum(axis=1).value_counts(ascending=False)
# Converting exports,imports and health spending percentages to absolute values.



ngo['exports'] = ngo['exports'] * ngo['gdpp']/100

ngo['imports'] = ngo['imports'] * ngo['gdpp']/100

ngo['health'] = ngo['health'] * ngo['gdpp']/100
ngo.head()
# Child Mortality Rate : Death of children under 5 years of age per 1000 live births

plt.figure(figsize = (30,5))

child_mort = ngo[['country','child_mort']].sort_values('child_mort', ascending = False)

ax = sns.barplot(x='country', y='child_mort', data= child_mort)

ax.set(xlabel = '', ylabel= 'Child Mortality Rate')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

child_mort_top10 = ngo[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)

ax = sns.barplot(x='country', y='child_mort', data= child_mort_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Child Mortality Rate')

plt.xticks(rotation=90)

plt.show()
# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same

plt.figure(figsize = (30,5))

total_fer = ngo[['country','total_fer']].sort_values('total_fer', ascending = False)

ax = sns.barplot(x='country', y='total_fer', data= total_fer)

ax.set(xlabel = '', ylabel= 'Fertility Rate')

plt.xticks(rotation=90)

plt.show()

plt.figure(figsize = (10,5))

total_fer_top10 = ngo[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)

ax = sns.barplot(x='country', y='total_fer', data= total_fer_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Fertility Rate')

plt.xticks(rotation=90)

plt.show()
# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same

plt.figure(figsize = (32,5))

life_expec = ngo[['country','life_expec']].sort_values('life_expec', ascending = True)

ax = sns.barplot(x='country', y='life_expec', data= life_expec)

ax.set(xlabel = '', ylabel= 'Life Expectancy')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

life_expec_bottom10 = ngo[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)

ax = sns.barplot(x='country', y='life_expec', data= life_expec_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Life Expectancy')

plt.xticks(rotation=90)

plt.show()
# Health :Total health spending as %age of Total GDP.

plt.figure(figsize = (32,5))

health = ngo[['country','health']].sort_values('health', ascending = True)

ax = sns.barplot(x='country', y='health', data= health)

ax.set(xlabel = '', ylabel= 'Health')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

health_bottom10 = ngo[['country','health']].sort_values('health', ascending = True).head(10)

ax = sns.barplot(x='country', y='health', data= health_bottom10)

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Health')

plt.xticks(rotation=90)

plt.show()
# The GDP per capita : Calculated as the Total GDP divided by the total population.

plt.figure(figsize = (32,5))

gdpp = ngo[['country','gdpp']].sort_values('gdpp', ascending = True)

ax = sns.barplot(x='country', y='gdpp', data= gdpp)

ax.set(xlabel = '', ylabel= 'GDP per capita')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

gdpp_bottom10 = ngo[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)

ax = sns.barplot(x='country', y='gdpp', data= gdpp_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'GDP per capita')

plt.xticks(rotation=90)

plt.show()
# Per capita Income : Net income per person

plt.figure(figsize = (32,5))

income = ngo[['country','income']].sort_values('income', ascending = True)

ax = sns.barplot(x='country', y='income', data=income)

ax.set(xlabel = '', ylabel= 'Per capita Income')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

income_bottom10 = ngo[['country','income']].sort_values('income', ascending = True).head(10)

ax = sns.barplot(x='country', y='income', data= income_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Per capita Income')

plt.xticks(rotation=90)

plt.show()
# Inflation: The measurement of the annual growth rate of the Total GDP

plt.figure(figsize = (32,5))

inflation = ngo[['country','inflation']].sort_values('inflation', ascending = False)

ax = sns.barplot(x='country', y='inflation', data= inflation)

ax.set(xlabel = '', ylabel= 'Inflation')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

inflation_top10 = ngo[['country','inflation']].sort_values('inflation', ascending = False).head(10)

ax = sns.barplot(x='country', y='inflation', data= inflation_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Inflation')

plt.xticks(rotation=90)

plt.show()
# Exports: Exports of goods and services. Given as %age of the Total GDP

plt.figure(figsize = (32,5))

exports = ngo[['country','exports']].sort_values('exports', ascending = True)

ax = sns.barplot(x='country', y='exports', data= exports)

ax.set(xlabel = '', ylabel= 'Exports')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

exports_bottom10 = ngo[['country','exports']].sort_values('exports', ascending = True).head(10)

ax = sns.barplot(x='country', y='exports', data= exports_bottom10)

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exports')

plt.xticks(rotation=90)

plt.show()
# Imports: Imports of goods and services. Given as %age of the Total GDP

plt.figure(figsize = (32,5))

imports = ngo[['country','imports']].sort_values('imports', ascending = True)

ax = sns.barplot(x='country', y='imports', data= imports)

ax.set(xlabel = '', ylabel= 'Imports')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

imports_bottom10 = ngo[['country','imports']].sort_values('imports', ascending = True).head(10)

ax = sns.barplot(x='country', y='imports', data= imports_bottom10)

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Imports')

plt.xticks(rotation=90)

plt.show()
fig, axs = plt.subplots(3,3,figsize = (18,18))



# Child Mortality Rate : Death of children under 5 years of age per 1000 live births



top5_child_mort = ngo[['country','child_mort']].sort_values('child_mort', ascending = False).head()

ax = sns.barplot(x='country', y='child_mort', data= top5_child_mort, ax = axs[0,0])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Child Mortality Rate')



# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same

top5_total_fer = ngo[['country','total_fer']].sort_values('total_fer', ascending = False).head()

ax = sns.barplot(x='country', y='total_fer', data= top5_total_fer, ax = axs[0,1])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Fertility Rate')



# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same



bottom5_life_expec = ngo[['country','life_expec']].sort_values('life_expec', ascending = True).head()

ax = sns.barplot(x='country', y='life_expec', data= bottom5_life_expec, ax = axs[0,2])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Life Expectancy')



# Health :Total health spending as %age of Total GDP.



bottom5_health = ngo[['country','health']].sort_values('health', ascending = True).head()

ax = sns.barplot(x='country', y='health', data= bottom5_health, ax = axs[1,0])

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Health')



# The GDP per capita : Calculated as the Total GDP divided by the total population.



bottom5_gdpp = ngo[['country','gdpp']].sort_values('gdpp', ascending = True).head()

ax = sns.barplot(x='country', y='gdpp', data= bottom5_gdpp, ax = axs[1,1])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'GDP per capita')



# Per capita Income : Net income per person



bottom5_income = ngo[['country','income']].sort_values('income', ascending = True).head()

ax = sns.barplot(x='country', y='income', data= bottom5_income, ax = axs[1,2])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Per capita Income')





# Inflation: The measurement of the annual growth rate of the Total GDP



top5_inflation = ngo[['country','inflation']].sort_values('inflation', ascending = False).head()

ax = sns.barplot(x='country', y='inflation', data= top5_inflation, ax = axs[2,0])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Inflation')





# Exports: Exports of goods and services. Given as %age of the Total GDP



bottom5_exports = ngo[['country','exports']].sort_values('exports', ascending = True).head()

ax = sns.barplot(x='country', y='exports', data= bottom5_exports, ax = axs[2,1])

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exports')





# Imports: Imports of goods and services. Given as %age of the Total GDP



bottom5_imports = ngo[['country','imports']].sort_values('imports', ascending = True).head()

ax = sns.barplot(x='country', y='imports', data= bottom5_imports, ax = axs[2,2])

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Imports')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation = 90)    

plt.tight_layout()

plt.savefig('EDA')

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (10, 10))

sns.heatmap(ngo.corr(), annot = True, cmap="rainbow")

plt.savefig('Correlation')

plt.show()
sns.pairplot(ngo,corner=True,diag_kind="kde")

plt.show()
# Data before Outlier Treatment 

ngo.describe()
f, axes = plt.subplots(3, 3, figsize=(20, 15))

s=sns.violinplot(y=ngo.child_mort,ax=axes[0, 0])

axes[0, 0].set_title('Child Mortality Rate')

s=sns.violinplot(y=ngo.exports,ax=axes[0, 1])

axes[0, 1].set_title('Exports')

s=sns.violinplot(y=ngo.health,ax=axes[0, 2])

axes[0, 2].set_title('Health')



s=sns.violinplot(y=ngo.imports,ax=axes[1, 0])

axes[1, 0].set_title('Imports')

s=sns.violinplot(y=ngo.income,ax=axes[1, 1])

axes[1, 1].set_title('Income per Person')

s=sns.violinplot(y=ngo.inflation,ax=axes[1, 2])

axes[1, 2].set_title('Inflation')



s=sns.violinplot(y=ngo.life_expec,ax=axes[2, 0])

axes[2, 0].set_title('Life Expectancy')

s=sns.violinplot(y=ngo.total_fer,ax=axes[2, 1])

axes[2, 1].set_title('Total Fertility')

s=sns.violinplot(y=ngo.gdpp,ax=axes[2, 2])

axes[2, 2].set_title('GDP per Capita')

s.get_figure().savefig('boxplot subplots.png')

plt.show()
plt.figure(figsize = (20,20))

features=['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']

for i in enumerate(features):

    plt.subplot(4,3,i[0]+1)

    sns.distplot(ngo[i[1]])
Q3 = ngo.exports.quantile(0.99)

Q1 = ngo.exports.quantile(0.01)

ngo['exports'][ngo['exports']<=Q1]=Q1

ngo['exports'][ngo['exports']>=Q3]=Q3
Q3 = ngo.imports.quantile(0.99)

Q1 = ngo.imports.quantile(0.01)

ngo['imports'][ngo['imports']<=Q1]=Q1

ngo['imports'][ngo['imports']>=Q3]=Q3
Q3 = ngo.health.quantile(0.99)

Q1 = ngo.health.quantile(0.01)

ngo['health'][ngo['health']<=Q1]=Q1

ngo['health'][ngo['health']>=Q3]=Q3
Q3 = ngo.income.quantile(0.99)

Q1 = ngo.income.quantile(0.01)

ngo['income'][ngo['income']<=Q1]=Q1

ngo['income'][ngo['income']>=Q3]=Q3
Q3 = ngo.inflation.quantile(0.99)

Q1 = ngo.inflation.quantile(0.01)

ngo['inflation'][ngo['inflation']<=Q1]=Q1

ngo['inflation'][ngo['inflation']>=Q3]=Q3
Q3 = ngo.life_expec.quantile(0.99)

Q1 = ngo.life_expec.quantile(0.01)

ngo['life_expec'][ngo['life_expec']<=Q1]=Q1

ngo['life_expec'][ngo['life_expec']>=Q3]=Q3
Q3 = ngo.child_mort.quantile(0.99)

Q1 = ngo.child_mort.quantile(0.01)

ngo['child_mort'][ngo['child_mort']<=Q1]=Q1

ngo['child_mort'][ngo['child_mort']>=Q3]=Q3
Q3 = ngo.total_fer.quantile(0.99)

Q1 = ngo.total_fer.quantile(0.01)

ngo['total_fer'][ngo['total_fer']<=Q1]=Q1

ngo['total_fer'][ngo['total_fer']>=Q3]=Q3
Q3 = ngo.gdpp.quantile(0.99)

Q1 = ngo.total_fer.quantile(0.01)

ngo['gdpp'][ngo['gdpp']<=Q1]=Q1

ngo['gdpp'][ngo['gdpp']>=Q3]=Q3
# Data sfter Outlier Treatment 

ngo.describe()
f, axes = plt.subplots(3, 3, figsize=(20, 15))

s=sns.violinplot(y=ngo.child_mort,ax=axes[0, 0])

axes[0, 0].set_title('Child Mortality Rate')

s=sns.violinplot(y=ngo.exports,ax=axes[0, 1])

axes[0, 1].set_title('Exports')

s=sns.violinplot(y=ngo.health,ax=axes[0, 2])

axes[0, 2].set_title('Health')



s=sns.violinplot(y=ngo.imports,ax=axes[1, 0])

axes[1, 0].set_title('Imports')

s=sns.violinplot(y=ngo.income,ax=axes[1, 1])

axes[1, 1].set_title('Income per Person')

s=sns.violinplot(y=ngo.inflation,ax=axes[1, 2])

axes[1, 2].set_title('Inflation')



s=sns.violinplot(y=ngo.life_expec,ax=axes[2, 0])

axes[2, 0].set_title('Life Expectancy')

s=sns.violinplot(y=ngo.total_fer,ax=axes[2, 1])

axes[2, 1].set_title('Total Fertility')

s=sns.violinplot(y=ngo.gdpp,ax=axes[2, 2])

axes[2, 2].set_title('GDP per Capita')

s.get_figure().savefig('boxplot subplots.png')

plt.show()
# Dropping Country field as final dataframe will only contain data columns



ngo_drop = ngo.copy()

country = ngo_drop.pop('country')
ngo_drop.head()
# Calculating Hopkins score to know whether the data is good for clustering or not.



def hopkins(X):

    d = X.shape[1]

    n = len(X)

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

 

    HS = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(HS):

        print(ujd, wjd)

        HS = 0

 

    return HS

# Hopkins score

Hopkins_score=round(hopkins(ngo_drop),2)
print("{} is a good Hopkins score for Clustering.".format(Hopkins_score))
# Standarisation technique for scaling

scaler = StandardScaler()

ngo_scaled = scaler.fit_transform(ngo_drop)
ngo_scaled
ngo_df1 = pd.DataFrame(ngo_scaled, columns = ['child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp'])

ngo_df1.head()
# Elbow curve method to find the ideal number of clusters.

clusters=list(range(2,8))

ssd = []

for num_clusters in clusters:

    model_clus = KMeans(n_clusters = num_clusters, max_iter=150,random_state= 50)

    model_clus.fit(ngo_df1)

    ssd.append(model_clus.inertia_)



plt.plot(clusters,ssd);
# Silhouette score analysis to find the ideal number of clusters for K-means clustering



range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state= 100)

    kmeans.fit(ngo_df1)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(ngo_df1, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
#K-means with k=3 clusters



cluster = KMeans(n_clusters=3, max_iter=150, random_state= 50)

cluster.fit(ngo_df1)
# Cluster labels



cluster.labels_
# Assign the label



ngo['Cluster_Id'] = cluster.labels_

ngo.head()
## Number of countries in each cluster

ngo.Cluster_Id.value_counts(ascending=True)
# Scatter plot on Original attributes to visualize the spread of the data



plt.figure(figsize = (20,15))

plt.subplot(3,1,1)

sns.scatterplot(x = 'income', y = 'child_mort',hue='Cluster_Id',data = ngo,legend='full',palette="Set1")

plt.subplot(3,1,2)

sns.scatterplot(x = 'gdpp', y = 'income',hue='Cluster_Id', data = ngo,legend='full',palette="Set1")

plt.subplot(3,1,3)

sns.scatterplot(x = 'child_mort', y = 'gdpp',hue='Cluster_Id', data=ngo,legend='full',palette="Set1")

plt.show()
 #Violin plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.violinplot(x = 'Cluster_Id', y = 'child_mort', data = ngo,ax=axes[0][0])

sns.violinplot(x = 'Cluster_Id', y = 'income', data = ngo,ax=axes[0][1])

sns.violinplot(x = 'Cluster_Id', y = 'inflation', data=ngo,ax=axes[1][0])

sns.violinplot(x = 'Cluster_Id', y = 'gdpp', data=ngo,ax=axes[1][1])

plt.show()
ngo[['gdpp','income','child_mort','Cluster_Id']].groupby('Cluster_Id').mean()
ax=ngo[['gdpp','child_mort','income','Cluster_Id']].groupby('Cluster_Id').mean().plot(kind = 'bar',figsize = (15,5))



for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.yscale('log')

plt.xticks(rotation=0)

plt.show();
ngo[ngo['Cluster_Id']==1].sort_values(by = ['child_mort','income','gdpp',], ascending = [False, True, True]).head()

# They are Developed countries as per UN & IMF
ngo[ngo['Cluster_Id']==2].sort_values(by = ['child_mort','income','gdpp',], ascending = [False, True, True]).head()



# They are Least developed countries as per UN & IMF
ngo[ngo['Cluster_Id']==0].sort_values(by = ['child_mort','income','gdpp',], ascending = [False, True, True]).head()

# They are Developing countries as per UN & IMF



FinalListbyKMean=ngo[ngo['Cluster_Id']==2].sort_values(by = ['child_mort','income','gdpp',], ascending = [False, True, True])

FinalListbyKMean['country']

FinalListbyKMean.reset_index(drop=True).country[:]
# BarPlot for Child Mortality of countries which are in need of aid

df_list_cm = pd.DataFrame(FinalListbyKMean.groupby(['country'])['child_mort'].mean().sort_values(ascending = False)).head()

ax=df_list_cm.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Country and Child Mortality')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("Child Mortality", fontsize = 12, fontweight = 'bold')

plt.show()
# BarPlot for Per Capita Income of countries which are in need of aid



df_list_in = pd.DataFrame(FinalListbyKMean.groupby(['country'])['income'].mean().sort_values(ascending = True)).head()

ax=df_list_in.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Country and Per Capita Income')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("Per Capita Income", fontsize = 12, fontweight = 'bold')

plt.show()
# BarPlot for GDP of countries which are in need of aid



df_list_gdp =pd.DataFrame(FinalListbyKMean.groupby(['country'])['gdpp'].mean().sort_values(ascending = True)).head()

ax=df_list_gdp.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.title('Country and GDP per capita')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("GDP per capita", fontsize = 12, fontweight = 'bold')

plt.show()
# Final countries list

FinalListbyKMean.reset_index(drop=True).country[:5]