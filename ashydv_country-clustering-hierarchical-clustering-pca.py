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

pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 50)



# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
data = pd.DataFrame(pd.read_csv('../input/Country-data.csv'))

data.head(5)
#checking duplicates

sum(data.duplicated(subset = 'country')) == 0

# No duplicate values
data.shape
data.info()
data.describe()
data.isnull().sum()
# No NULL values are observed.
# We will have a look on the lowest 10 countries for each factor.
fig, axs = plt.subplots(3,3,figsize = (15,15))



# Child Mortality Rate : Death of children under 5 years of age per 1000 live births



top10_child_mort = data[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)

plt1 = sns.barplot(x='country', y='child_mort', data= top10_child_mort, ax = axs[0,0])

plt1.set(xlabel = '', ylabel= 'Child Mortality Rate')



# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same

top10_total_fer = data[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)

plt1 = sns.barplot(x='country', y='total_fer', data= top10_total_fer, ax = axs[0,1])

plt1.set(xlabel = '', ylabel= 'Fertility Rate')



# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same



bottom10_life_expec = data[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='life_expec', data= bottom10_life_expec, ax = axs[0,2])

plt1.set(xlabel = '', ylabel= 'Life Expectancy')



# Health :Total health spending as %age of Total GDP.



bottom10_health = data[['country','health']].sort_values('health', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='health', data= bottom10_health, ax = axs[1,0])

plt1.set(xlabel = '', ylabel= 'Health')



# The GDP per capita : Calculated as the Total GDP divided by the total population.



bottom10_gdpp = data[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='gdpp', data= bottom10_gdpp, ax = axs[1,1])

plt1.set(xlabel = '', ylabel= 'GDP per capita')



# Per capita Income : Net income per person



bottom10_income = data[['country','income']].sort_values('income', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='income', data= bottom10_income, ax = axs[1,2])

plt1.set(xlabel = '', ylabel= 'Per capita Income')





# Inflation: The measurement of the annual growth rate of the Total GDP



top10_inflation = data[['country','inflation']].sort_values('inflation', ascending = False).head(10)

plt1 = sns.barplot(x='country', y='inflation', data= top10_inflation, ax = axs[2,0])

plt1.set(xlabel = '', ylabel= 'Inflation')





# Exports: Exports of goods and services. Given as %age of the Total GDP



bottom10_exports = data[['country','exports']].sort_values('exports', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='exports', data= bottom10_exports, ax = axs[2,1])

plt1.set(xlabel = '', ylabel= 'Exports')





# Imports: Imports of goods and services. Given as %age of the Total GDP



bottom10_imports = data[['country','imports']].sort_values('imports', ascending = True).head(10)

plt1 = sns.barplot(x='country', y='imports', data= bottom10_imports, ax = axs[2,2])

plt1.set(xlabel = '', ylabel= 'Imports')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation = 90)

    

plt.tight_layout()

plt.savefig('eda')

plt.show()

    

# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (16, 10))

sns.heatmap(data.corr(), annot = True, cmap="YlGnBu")

plt.savefig('corrplot')

plt.show()
# We can see there is high correlation between some variables, we will use PCA to solve this issue.
# We will see how values in each columns are distributed using boxplot
fig, axs = plt.subplots(3,3, figsize = (15,7.5))

plt1 = sns.boxplot(data['child_mort'], ax = axs[0,0])

plt2 = sns.boxplot(data['health'], ax = axs[0,1])

plt3 = sns.boxplot(data['life_expec'], ax = axs[0,2])

plt4 = sns.boxplot(data['total_fer'], ax = axs[1,0])

plt5 = sns.boxplot(data['income'], ax = axs[1,1])

plt6 = sns.boxplot(data['inflation'], ax = axs[1,2])

plt7 = sns.boxplot(data['gdpp'], ax = axs[2,0])

plt8 = sns.boxplot(data['imports'], ax = axs[2,1])

plt9 = sns.boxplot(data['exports'], ax = axs[2,2])





plt.tight_layout()

data.describe()
# Before manipulating data, we will save one copy of orignal data.

data_help = data.copy()

data_help.head()
# As we can see there are a number of outliers in the data.



# Keeping in mind we need to identify backward countries based on socio economic and health factors.

# We will cap the outliers to values accordingly for analysis.



percentiles = data_help['child_mort'].quantile([0.05,0.95]).values

data_help['child_mort'][data_help['child_mort'] <= percentiles[0]] = percentiles[0]

data_help['child_mort'][data_help['child_mort'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['health'].quantile([0.05,0.95]).values

data_help['health'][data_help['health'] <= percentiles[0]] = percentiles[0]

data_help['health'][data_help['health'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['life_expec'].quantile([0.05,0.95]).values

data_help['life_expec'][data_help['life_expec'] <= percentiles[0]] = percentiles[0]

data_help['life_expec'][data_help['life_expec'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['total_fer'].quantile([0.05,0.95]).values

data_help['total_fer'][data_help['total_fer'] <= percentiles[0]] = percentiles[0]

data_help['total_fer'][data_help['total_fer'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['income'].quantile([0.05,0.95]).values

data_help['income'][data_help['income'] <= percentiles[0]] = percentiles[0]

data_help['income'][data_help['income'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['inflation'].quantile([0.05,0.95]).values

data_help['inflation'][data_help['inflation'] <= percentiles[0]] = percentiles[0]

data_help['inflation'][data_help['inflation'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['gdpp'].quantile([0.05,0.95]).values

data_help['gdpp'][data_help['gdpp'] <= percentiles[0]] = percentiles[0]

data_help['gdpp'][data_help['gdpp'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['imports'].quantile([0.05,0.95]).values

data_help['imports'][data_help['imports'] <= percentiles[0]] = percentiles[0]

data_help['imports'][data_help['imports'] >= percentiles[1]] = percentiles[1]



percentiles = data_help['exports'].quantile([0.05,0.95]).values

data_help['exports'][data_help['exports'] <= percentiles[0]] = percentiles[0]

data_help['exports'][data_help['exports'] >= percentiles[1]] = percentiles[1]
fig, axs = plt.subplots(3,3, figsize = (15,7.5))



plt1 = sns.boxplot(data_help['child_mort'], ax = axs[0,0])

plt2 = sns.boxplot(data_help['health'], ax = axs[0,1])

plt3 = sns.boxplot(data_help['life_expec'], ax = axs[0,2])

plt4 = sns.boxplot(data_help['total_fer'], ax = axs[1,0])

plt5 = sns.boxplot(data_help['income'], ax = axs[1,1])

plt6 = sns.boxplot(data_help['inflation'], ax = axs[1,2])

plt7 = sns.boxplot(data_help['gdpp'], ax = axs[2,0])

plt8 = sns.boxplot(data_help['imports'], ax = axs[2,1])

plt9 = sns.boxplot(data_help['exports'], ax = axs[2,2])



plt.tight_layout()
# Import the StandardScaler()

from sklearn.preprocessing import StandardScaler



# Create a scaling object

scaler = StandardScaler()



# Create a list of the variables that you need to scale

varlist = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

# Scale these variables using 'fit_transform'

data_help[varlist] = scaler.fit_transform(data_help[varlist])
#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)
# Putting feature variable to X

X = data_help.drop(['country'],axis=1)



# Putting response variable to y

y = data_help['country']
#Doing the PCA on the train data

pca.fit(X)
pca.components_
colnames = list(X.columns)

pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})

pcs_df.head()
%matplotlib inline

fig = plt.figure(figsize = (8,8))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
pca.explained_variance_ratio_
#Making the screeplot - plotting the cumulative variance against the number of components

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.savefig('pca_no')

plt.show()
#Using incremental PCA for efficiency - saves a lot of time on larger datasets

from sklearn.decomposition import IncrementalPCA

pca_final = IncrementalPCA(n_components=4)
df_pca = pca_final.fit_transform(X)

df_pca.shape
df_pca = pd.DataFrame(df_pca)

df_pca.head()
#creating correlation matrix for the principal components

corrmat = np.corrcoef(df_pca.transpose())
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (10,5))

sns.heatmap(corrmat,annot = True)
# To perform KMeans clustering 

from sklearn.cluster import KMeans
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
hopkins(df_pca)
#  high tendency to cluster.
mergings = linkage(df_pca, method = "complete", metric='euclidean')

dendrogram(mergings)

plt.show()
# Looking at the dedrogram it is observed that cutting it at n = 5 is most optimum.
clusterCut = pd.Series(cut_tree(mergings, n_clusters = 5).reshape(-1,))

df_pca_hc = pd.concat([df_pca, clusterCut], axis=1)

df_pca_hc.columns = ["PC1","PC2","PC3","PC4","ClusterID"]

df_pca_hc.head()

pca_cluster_hc = pd.concat([data_help['country'],df_pca_hc], axis=1, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True)

pca_cluster_hc.head()
clustered_data_hc = pca_cluster_hc[['country','ClusterID']].merge(data, on = 'country')

clustered_data_hc.head()
hc_clusters_child_mort = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).child_mort.mean())

hc_clusters_exports = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).exports.mean())

hc_clusters_health = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).health.mean())

hc_clusters_imports = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).imports.mean())

hc_clusters_income = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).income.mean())

hc_clusters_inflation = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).inflation.mean())

hc_clusters_life_expec = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).life_expec.mean())

hc_clusters_total_fer = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).total_fer.mean())

hc_clusters_gdpp = 	pd.DataFrame(clustered_data_hc.groupby(["ClusterID"]).gdpp.mean())
df = pd.concat([pd.Series(list(range(0,5))), hc_clusters_child_mort,hc_clusters_exports, hc_clusters_health, hc_clusters_imports,

               hc_clusters_income, hc_clusters_inflation, hc_clusters_life_expec,hc_clusters_total_fer,hc_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort_mean", "exports_mean", "health_mean", "imports_mean", "income_mean", "inflation_mean",

               "life_expec_mean", "total_fer_mean", "gdpp_mean"]

df
fig, axs = plt.subplots(3,3,figsize = (15,15))



sns.barplot(x=df.ClusterID, y=df.child_mort_mean, ax = axs[0,0])

sns.barplot(x=df.ClusterID, y=df.exports_mean, ax = axs[0,1])

sns.barplot(x=df.ClusterID, y=df.health_mean, ax = axs[0,2])

sns.barplot(x=df.ClusterID, y=df.imports_mean, ax = axs[1,0])

sns.barplot(x=df.ClusterID, y=df.income_mean, ax = axs[1,1])

sns.barplot(x=df.ClusterID, y=df.life_expec_mean, ax = axs[1,2])

sns.barplot(x=df.ClusterID, y=df.inflation_mean, ax = axs[2,0])

sns.barplot(x=df.ClusterID, y=df.total_fer_mean, ax = axs[2,1])

sns.barplot(x=df.ClusterID, y=df.gdpp_mean, ax = axs[2,2])

plt.tight_layout()
clustered_data_hc[clustered_data_hc.ClusterID == 0].country.values