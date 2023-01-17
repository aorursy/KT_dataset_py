import pandas as pd

pd.set_option('display.max_column',None)

pd.set_option('display.max_rows',None)

pd.set_option('display.max_seq_items',None)

pd.set_option('display.max_colwidth', 500)

pd.set_option('expand_frame_repr', True)

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

 

import warnings

warnings.filterwarnings('ignore')
#Columns description view of data source file : "data-dictionary.csv"

data_dic = pd.read_csv("../input/data-dictionary.csv")

data_dic
df=pd.read_csv("../input/Country-data.csv")
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.dtypes.value_counts()
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
df.isnull().sum()*100/df.shape[0]
df.loc[df.duplicated()]
df.columns
numerical = df.select_dtypes(include=[np.number]).columns.tolist()

numerical
categorical = df.select_dtypes(include=[np.object]).columns.tolist()

categorical
pd.plotting.scatter_matrix(df, alpha=0.7, figsize=(15, 15), diagonal='kde')

plt.title('Visualize the dataset ')

plt.show()

plt.figure(figsize = (15,8))  

result_corr=df.corr(method='pearson')

sns.heatmap(result_corr[(result_corr >= 0.5) | (result_corr <= -0.4)], 

            cmap='YlGnBu', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);

plt.title('Correlation Matrix ')
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [10, 5]})

sns.distplot(

    df['child_mort'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}

).set(xlabel='Distribution Child Mortality Rate', ylabel='Count',title='Distribution of Child Mortality Rate');

 
cols=['exports',

 'health',

 'imports',

 'income',

 'inflation',

 'life_expec',

 'total_fer',

 'gdpp']



df[cols].hist(bins=15, figsize=(15, 6), layout=(2, 4) );

def column_scatter(df,xcol,ycol,nrows,ncols,figno):

    '''

    * column_scatter will plot scatter graph based on the parameters.

    * df   : dataframe

    * xcol : xcol/x-axis variable

    * ycol : ycol/y-axis variable

    * nrows: no. of rows for sub-plots

    * ncols: no. of cols for sub-plots

    

    '''

    plt.subplot(nrows,ncols,figno)

    plt.scatter(df[xcol],df[ycol])

    plt.title(xcol+' vs ' + ycol)

    plt.ylabel(ycol)

    plt.xlabel(xcol)

    plt.tight_layout() 

    

cols=['exports', 

 'health',

 'imports',

 'income',

 'inflation',

 'life_expec',

 'total_fer',

 'child_mort']

plt.figure(figsize=(15,10))



nrows,ncols=2,4

count=nrows*ncols

for col in cols:

    column_scatter(df,col ,'gdpp',nrows,ncols,count)

    count-=1   

# Child Mortality Rate : Death of children under 5 years of age per 1000 live births

result = df[['country','child_mort']].sort_values('child_mort', ascending = False).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("Child Mortality Rate")

Plt.set_title("Child Mortality Rate vs Country")

result.rename(columns={'country':'Country Name','child_mort':'Child Mortality Rate'},inplace=True)

 

result
# Imports: Imports of goods and services. Given as %age of the Total GDP



result = df[['country','imports']].sort_values('imports', ascending = True).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("Imports")

Plt.set_title("Imports vs Country")

result.rename(columns={'country':'Country Name','imports':'Imports of goods and services.'},inplace=True)

result
# Exports: Exports of goods and services. Given as %age of the Total GDP



result = df[['country','exports']].sort_values('exports', ascending = True).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("exports")

Plt.set_title("Exports vs Country")

result.rename(columns={'country':'Country Name','exports':'Exports of goods and services.'},inplace=True)

result
# Health :Total health spending as %age of Total GDP.

result = df[['country','health']].sort_values('health', ascending = True).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("Health")

Plt.set_title("Health vs Country")

result.rename(columns={'country':'Country Name','health':'Total Health Spending as %ge of Total GDP'},inplace=True)

result
# Per capita Income : Net income per person

result = df[['country','income']].sort_values('income', ascending = True).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("Income")

Plt.set_title("Income vs Country")

result.rename(columns={'country':'Country Name','income':'Per capita Income : Net income per person'},inplace=True)

result
# Inflation: The measurement of the annual growth rate of the Total GDP

result = df[['country','inflation']].sort_values('inflation', ascending = False).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

 

Plt.set_ylabel("Inflation Rate")

Plt.set_title("Inflation Rate vs Country")

result.rename(columns={'country':'Country Name','income':'Inflation: The measurement of the annual growth rate of the Total GDP'},inplace=True)

result
#Life Expectancy: The average number of years a new born child would live if the current mortality patterns 

# are to remain same

result = df[['country','life_expec']].sort_values('life_expec', ascending = True).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("Life Expectancy")

Plt.set_title("Life Expectancy vs Country")

result.rename(columns={'country':'Country Name','life_expec':'Life Expectancy'},inplace=True)

result
# Fertility Rate: The number of children that would be born to each woman if the current age-fertility 

# rates remain the same



result = df[['country','total_fer']].sort_values('total_fer', ascending = False).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("Fertility Rate")

Plt.set_title("Fertility Rate vs Country")

result.rename(columns={'country':'Country Name','total_fer':'Fertility Rate'},inplace=True)

result
# The GDP per capita : Calculated as the Total GDP divided by the total population.



result = df[['country','gdpp']].sort_values('gdpp', ascending = True).head(5)

Plt = result.plot(x = 'country', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_ylabel("GDP per capita")

Plt.set_title("GDP per capitavs Country")

result.rename(columns={'country':'Country Name','gdpp':'GDP per capita'},inplace=True)

result
df.describe(percentiles=[.25,.35,.75,.90,.95,.99])
result=df[numerical]

nrow,ncol = 3,3

fig, axes = plt.subplots(nrow, ncol,figsize = (16,8))

for col,axis in zip(numerical,axes.flat):

    sns.boxplot(result[col], color='blue',ax =axis )

plt.tight_layout()
dfs = df.copy()

dfs.head()
#Use clip() function

popped_col = dfs.pop("country") 

minPercentile = 0.05

maxPercentile = 0.95

dfs=dfs.clip(dfs.quantile(minPercentile), dfs.quantile(maxPercentile), axis=1)

dfs.head()

 
nrow,ncol = 3,3

fig, axes = plt.subplots(nrow, ncol,figsize = (16,8))

for col,axis in zip(dfs.columns.tolist(),axes.flat):

    sns.boxplot(dfs[col], color='blue',ax =axis)

plt.tight_layout()
dfs["country"]= popped_col

dfs=dfs.reindex([ 'country','child_mort', 'exports', 'health', 'imports', 'income', 'inflation',

       'life_expec', 'total_fer', 'gdpp'], axis=1)

dfs.head()
dfs.shape
plt.figure(figsize = (16,10))  

sns.heatmap(dfs.corr(method='pearson'), annot = True, cmap="RdBu")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

dfs[cols] = scaler.fit_transform(dfs[cols])

dfs.head()
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
X = dfs.drop(['country'],axis=1)

y = dfs['country']
pca.fit(X)
X.shape
explained_variance = pca.explained_variance_ratio_

explained_variance
components = pd.DataFrame(pca.components_, columns = X.columns)

components
colnames = list(X.columns)

pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})

pcs_df.head()
fig = plt.figure(figsize = (10,8))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
pca.explained_variance_ratio_

fig = plt.figure(figsize = (10,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.title('Cumulative Cariance against the Number of Components')

plt.xlabel('Number of Components')

plt.ylabel('Cumulative Explained Variance')

plt.show()
#Using incremental PCA for efficiency. 

from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=4)
dfs_pca = pca_final.fit_transform(X)

dfs_pca.shape
dfs_pca = pd.DataFrame(data = dfs_pca

             , columns = ['PC1','PC2','PC3','PC4'])

dfs_pca.head()
#creating correlation matrix for the principal components

corrmat = np.corrcoef(dfs_pca.transpose())
#plotting the correlation matrix

plt.figure(figsize = (12,6))

plt.title('Correlation Matrix for the Principal Components')

sns.heatmap(corrmat,annot = True,cmap="RdBu")

plt.show()
# Import Kmeans

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

 

from math import isnan

 

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

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H

print("dfs_no_pca: ", hopkins(X))

print("dfs_pca   : ", hopkins(dfs_pca))

# import silhouette lib from sklearn 

from sklearn.metrics import silhouette_score

sse = []

for num_clusters in range(2, 21):

    kmeans = KMeans(n_clusters=num_clusters, n_init = 50).fit(dfs_pca)

    sse.append([num_clusters, silhouette_score(dfs_pca, kmeans.labels_)])

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(dfs_pca, cluster_labels)

    print("For n_clusters={0}, The average silhouette_score is {1}".format(num_clusters, round(silhouette_avg,2) ))

    

plt.figure(figsize = (8,6))

sns.pointplot(pd.DataFrame(sse)[0], y = pd.DataFrame(sse)[1])

plt.xlabel('No. of clusters')

plt.ylabel('Silhouette score')

# sum of squared distances

ssd = []

for num_clusters in list(range(1,21)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(dfs_pca)

    ssd.append([num_clusters, model_clus.inertia_])

    

plt.figure(figsize = (8,6))

sns.pointplot(x = pd.DataFrame(ssd)[0], y = pd.DataFrame(ssd)[1]) 



plt.xlabel('No. of Clusters')

plt.ylabel('Sum of Squared Errors')

kmeans = KMeans(n_clusters=4, max_iter=50)

kmeans.fit(dfs_pca)
kmeans.labels_
dfkmeans=df.copy()

dfkmeans['cluster_id_kmeans'] = kmeans.labels_

dfkmeans.head()
# Call Seaborn's pairplot to visualize our KMeans clustering to understand the clustered data

sns.pairplot(dfkmeans, hue='cluster_id_kmeans', palette= 'Dark2', diag_kind='kde',size=1.85)

plt.show()
dfs_pca['cluster_id_kmeans'] = kmeans.labels_

dfs_pca.head()
ax = dfs_pca.plot(kind='scatter', x='PC1', y='PC2',

                           figsize=(10,8) ,c='cluster_id_kmeans',  s=130, linewidth=0,

            cmap=plt.cm.get_cmap('Spectral',25) )
cmap = plt.get_cmap('Spectral')

ax = dfkmeans.plot(kind='scatter', x='child_mort', y='exports',

                           figsize=(10,8) ,c='cluster_id_kmeans', s=130, linewidth=0, 

          colormap=cmap ) 
kmeans_mean_cluster = pd.DataFrame(round(dfkmeans.groupby([  'cluster_id_kmeans']).mean(),1))

kmeans_mean_cluster = kmeans_mean_cluster.reset_index()

kmeans_mean_cluster
#plot child morality rate,income and GDP per capita for cluster id.



plt.figure(figsize=(20,8))



plt.subplot(1,3,1)

sns.barplot(x='cluster_id_kmeans', y='child_mort', data=kmeans_mean_cluster) 

plt.subplot(1,3,2)

sns.barplot(x='cluster_id_kmeans', y='income', data=kmeans_mean_cluster)

plt.subplot(1,3,3)

sns.barplot(x='cluster_id_kmeans', y='gdpp', data=kmeans_mean_cluster)

plt.show()
kmeans_mean_cluster.columns
group_df = dfkmeans.groupby("cluster_id_kmeans")["child_mort","gdpp","income","exports"].mean().sort_values("child_mort",ascending=False)

tempdf = pd.DataFrame(group_df).reset_index()

dfkmeans[dfkmeans.cluster_id_kmeans == tempdf.cluster_id_kmeans[0]].country.values
dfkmeans_cluster = dfkmeans.loc[(dfkmeans['cluster_id_kmeans']==tempdf.cluster_id_kmeans[0])]

kmeans_cluster = dfkmeans[['country','child_mort', 'gdpp', 'income']]

kmeans_cluster_child_mort= dfkmeans.nlargest(5, ['child_mort'])

kmeans_cluster_gdp = dfkmeans.nsmallest(5, ['gdpp'])

kmeans_cluster_income = dfkmeans.nsmallest(5, ['income'])
plt.figure(figsize=(20,8))



plt.subplot(1,3,1)

plot=sns.barplot(y='country', x='child_mort', data=kmeans_cluster_child_mort ) 

plt.setp(plot.get_xticklabels(), rotation=90)



plt.subplot(1,3,2)

plot=sns.barplot(y='country', x='gdpp', data=kmeans_cluster_gdp)

plt.setp(plot.get_xticklabels(), rotation=90)

plt.subplot(1,3,3)

plot=sns.barplot(y='country', x='income', data=kmeans_cluster_income)

plt.setp(plot.get_xticklabels(), rotation=90)

fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

plt.show()

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

dfs_pca.head()
dfs.head()
plt.figure(figsize=(12,8))

mergings_s = linkage(dfs_pca, method = "single", metric='euclidean')

dendrogram(mergings_s, labels=dfs_pca.index, leaf_rotation=90, leaf_font_size=6)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: Single linkage Method",fontsize=20)

plt.show()
plt.figure(figsize=(12,8))

mergings_s = linkage(dfs_pca, method = "single", metric='euclidean')

dendrogram(mergings_s, leaf_rotation=90, p=50, truncate_mode='lastp')

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: Truncated Single linkage Method",fontsize=20)

plt.show()
plt.figure(figsize=(12,8))

mergings_c = linkage(dfs_pca, method = "complete", metric='euclidean')

dendrogram(mergings_c, labels=dfs_pca.index, leaf_rotation=90, leaf_font_size=6)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: Complete Linkage Method",fontsize=20)

plt.show()
dfs_pca.head()
plt.figure(figsize=(12,8))

mergings_c = linkage(dfs_pca, method = "complete", metric='euclidean')

dendrogram(mergings_c, labels=dfs_pca.index, leaf_rotation=90, leaf_font_size=6)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: Complete Linkage Method",fontsize=20)

plt.axhline(y=5.8, c='k',linestyle='--')



plt.show()
dfs_pca.columns
clusterCut = pd.Series(cut_tree(mergings_c, n_clusters = 5).reshape(-1,))

dfs_pca_hc = pd.concat([dfs_pca, clusterCut], axis=1)

dfs_pca_hc.columns = ["PC1","PC2","PC3","PC4","cluster_id_kmeans","cluster_id_hc" ]

dfs_pca_hc.head()

pca_cluster_hc = pd.concat([dfs['country'],dfs_pca_hc], axis=1, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True)

pca_cluster_hc.head()
cmap = plt.get_cmap('Spectral')

ax = pca_cluster_hc.plot(kind='scatter', x='PC2', y='PC1',

                           figsize=(15,10) ,c='cluster_id_hc', s=130, linewidth=0, colormap=cmap ) 

          

for i, country in enumerate(pca_cluster_hc.country):

    ax.annotate(country, (pca_cluster_hc.iloc[i].PC2, pca_cluster_hc.iloc[i].PC1),

                xytext=(1,10),

            textcoords='offset points', ha='center', va='bottom'

             ,rotation=90)
clustered_data_hc = pca_cluster_hc[['country','cluster_id_hc']].merge(df, on = 'country')

clustered_data_hc.head()
cmap = plt.get_cmap('Spectral')

 

ax = clustered_data_hc.plot(kind='scatter', x='child_mort', y='exports',

                           figsize=(15,10) ,c='cluster_id_hc', s=130, linewidth=0, 

          colormap=cmap ) 

 

for i, country in enumerate(clustered_data_hc.country):

    ax.annotate(country, (clustered_data_hc.iloc[i].child_mort, clustered_data_hc.iloc[i].exports),

                xytext=(1,10),

            textcoords='offset points', ha='center', va='bottom')
hc_clusters_child_mort = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).child_mort.mean())

hc_clusters_exports = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).exports.mean())

hc_clusters_health = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).health.mean())

hc_clusters_imports = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).imports.mean())

hc_clusters_income = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).income.mean())

hc_clusters_inflation = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).inflation.mean())

hc_clusters_life_expec = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).life_expec.mean())

hc_clusters_total_fer = pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).total_fer.mean())

hc_clusters_gdpp = 	pd.DataFrame(clustered_data_hc.groupby(["cluster_id_hc"]).gdpp.mean())

df_hc = pd.concat([pd.Series(list(range(0,5))),

                   hc_clusters_child_mort,hc_clusters_exports, 

                   hc_clusters_health, hc_clusters_imports,

                   hc_clusters_income, 

                   hc_clusters_inflation,

                   hc_clusters_life_expec,

                   hc_clusters_total_fer,

                   hc_clusters_gdpp], axis=1)

 

df_hc.columns = ["cluster_id_hc", "child_mort_mean",

                 "exports_mean", "health_mean",

                 "imports_mean", "income_mean", 

                 "inflation_mean","life_expec_mean",

                  "total_fer_mean", 

                 "gdpp_mean"]

df_hc.head()

numerical=[ "child_mort_mean","income_mean","gdpp_mean"]

        

nrow,ncol = 1,3

fig, axes = plt.subplots(nrow, ncol,figsize = (18,8))

for col,axis in zip(numerical,axes.flat):

    sns.barplot(x=df_hc.cluster_id_hc, y=df_hc[col],ax =axis)

     

plt.tight_layout()

group_df = clustered_data_hc.groupby("cluster_id_hc")["child_mort","gdpp","income","exports"].mean().sort_values("child_mort",ascending=False)

tempdf = pd.DataFrame(group_df).reset_index()

clustered_data_hc[clustered_data_hc.cluster_id_hc == tempdf.cluster_id_hc[0]].country.values
clustered_data_hc = clustered_data_hc.loc[(clustered_data_hc['cluster_id_hc']==tempdf.cluster_id_hc[0])]

clustered_data_hc = clustered_data_hc[['country','child_mort', 'gdpp', 'income']]

clustered_data_hc0_child_mort= clustered_data_hc.nlargest(5, ['child_mort'])

clustered_data_hc0_gdpp = clustered_data_hc.nsmallest(5, ['gdpp'])

clustered_data_hc0_income = clustered_data_hc.nsmallest(5, ['income'])
plt.figure(figsize=(18,6))



plt.subplot(1,3,1)

plot=sns.barplot(x='country', y='child_mort', data=clustered_data_hc0_child_mort ) 

plt.setp(plot.get_xticklabels(), rotation=90)



plt.subplot(1,3,2)

plot=sns.barplot(x='country', y='gdpp', data=clustered_data_hc0_gdpp)

plt.setp(plot.get_xticklabels(), rotation=90)

plt.subplot(1,3,3)

plot=sns.barplot(x='country', y='income', data=clustered_data_hc0_income)

plt.setp(plot.get_xticklabels(), rotation=90)

fig.tight_layout()  

plt.show() 
