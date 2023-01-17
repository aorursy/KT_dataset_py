# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# Importing libraries required for analysis



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

# To perform KMeans clustering 

from sklearn.cluster import KMeans

# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data_country_filepath='../input/datadictionary-countrydata/Country-data.csv'

data_dict_filepath='../input/datadictionary-countrydata/data-dictionary.csv'



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''

Assignment on Clustering and PCA

### `Selection of Countries using socio-economic and health factors`



## Index Area

1. [Business Objective](#Business-Objective)

2. [Loading Dataset](#Loading-Dataset)

3. [Cleaning Dataset](#Cleaning-Dataset)

4. [Exploratory Data Analysis](#Exploratory-Data-Analysis)

5. [Data Preparation for PCA and Clustering](#Data-Preparation-for-PCA-and-Clustering) 

    * [Feature Standardization or Normalization](#Feature-Standardization-or-Normalization)





6. [Selection of Principal Components using PCA](#Selection-of-Principal-Components-using-PCA)

7. [Outlier Analysis and Treatment](#Outlier-Analysis-and-Treatment)

8. [Visualizing Principal components](#Visualizing-Principal-components)

9. [Pre K-Means Clustering Analysis](#Pre-K-Means-Clustering-Analysis)

    * [Hopkins Analysis](#Hopkins-Analysis)

    * [Silhouette and Elbow Analysis](#Silhouette-and-Elbow-Analysis)





10. [K Means with K as 5](#K-Means-with-K-as-5)

11. [K Means with K as 6](#K-Means-with-K-as-6)

12. [K Means with K as 7](#K-Means-with-K-as-7)

13. [Hierarchical Clustering Single Method](#Hierarchical-Clustering-Single-Method)

14. [Hierarchical Clustering Complete Method](#Hierarchical-Clustering-Complete-Method)

15. [Combined list of countries from K-Means and Hierarchical Clustering](#Combined-list-of-countries-from-K-Means-and-Hierarchical-Clustering)

16. [Analysis of Outlier that were dropped](#Analysis-of-Outlier-that-were-dropped)

17. [Combining, Visualizing and Summarizing Results](#Combining,-Visualizing-and-Summarizing-Results)



----



## Business Objective



We need to help the CEO of HELP International, an international 

humanitarian NGO, in the process of selection of countries they should focus on.

 The selection will be based on some socio-economic and health factors

 that determine the overall development of the country.



In other words, it is required to cluster the countries 

by the factors mentioned above and then present our solution and recommendations

 to the CEO. We are also supposed to use dimensionality reduction using PCA 

 to get the visualizations of the clusters in a 2-D form.



----



## Loading Dataset

'''

country_data=pd.read_csv(data_country_filepath)

data_dict=pd.read_csv(data_dict_filepath)

print(country_data.head())

print(data_dict.head())


# Checking version of imports to refer version specific documentation

f'NumPy version: {np.__version__} | Pandas version: {pd.__version__} |Matplotlib version: {mpl.__version__} | Seaborn version: {sns.__version__}'





# Changing default display options for better visibility of data

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("display.max_colwidth",255)



# Displaying data from 'data-dictionary.csv' to be used for reference later

data_dict.head()



# Displaying some data from 'Country-data.csv'

country_data.head()





# Getting deatils for data dictionary

print(f'\nShape of dataframe:\n{country_data.shape}')

print(f'\nCount of dataframe:\n{country_data.count()}')

print(f'\nColumns in dataframe:\n{country_data.columns}')

print(f'\nDescription of dataframe:\n{country_data.describe()}')

print(f'\nNull values in dataframe:\n{country_data.isnull().sum()}')





# **Summary**:

# 1. There are 167 rows and 10 columns in dataframe

# 2. These 10 columns comprises of measures (numeric value) except the contry name, which anyways is going to be the output column

# 3. The dataset is clean, i.e. no missing/null values


# ## Cleaning Dataset



# Dropping duplicate values in dataset, if exist

country_data = country_data.drop_duplicates()

print(f'\nShape of dataframe:\n{country_data.shape}')





# Getting deatils for country_data dataset

print(f'\nShape of dataframe:\n{country_data.shape}')

print(f'\nCount of dataframe:\n{country_data.count()}')

print(f'\nDatatypes of dataframe:\n{country_data.dtypes}')

print(f'\nColumns in dataframe:\n{country_data.columns}')

print(f'\nDescription of dataframe:\n{country_data.describe()}')

print(f'\nNull values in dataframe:\n{country_data.isnull().sum()}')



# Separating numerical and categorical fields for analysisng them separately.



# OUTCOME COLUMN: 'country'



# NUMERICAL COLUMNS: 'child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp'

num_col = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']



# CATEGORICAL COLMNS: None





# **Note**: We no not have any categorical variable, thus we will skip dummification of categorical variables.



# ----
# ## Exploratory Data Analysis



# Function for Analysis and visualise of Numerical columns independently (Boxplot)

def univariate_analysis(i,col_i):

    plt.figure(i, figsize=(5, 5))

    sns.set(style="whitegrid",font_scale=.75)

    ax = sns.boxplot(y=col_i, data=country_data[[col_i]])

    ax = sns.swarmplot(y=col_i, data=country_data[[col_i]], color=".25")

    ax.set(ylabel="", xlabel=f"Distribution of {col_i}")

           

# Function for Analysis of Numerical column w.r.t. Country (Barplot)

def bivariate_analysis(i,col_i,col_o):

    plt.figure(i, figsize=(10, 30))

    sns.set(style="whitegrid",font_scale=.75)

    ax = sns.barplot(x=col_i, y=col_o, data=country_data[[col_i,col_o]], color="b")

    ax.set(ylabel="", xlabel=f"{col_i} [ {data_dict[data_dict['Column Name'].str.upper()==col_i.upper()]['Description'].to_string()[5:]} ] in each {col_o}")



# NUMERICAL COLUMN ANALYSIS (Barplot)

for i, col in enumerate(num_col):

    univariate_analysis(i, col)

# NUMERICAL COLUMN ANALYSIS (Barplot)

for i, col in enumerate(num_col):

    bivariate_analysis(i, col, 'country')





# NUMERICAL COLUMN HEATMAP (Co-relation Analysis)

plt.figure(i, figsize=(10, 10))

sns.set(style="whitegrid",font_scale=1.25)

sns.heatmap(country_data[num_col].corr(), annot=True, linewidths=2, square=True, cmap="YlGnBu")





# **Note**: Our initial columns are highly co-related and hence, not good to be used directly for model building. 

# We will use PCA to find good set of Principal Components which are not co-related.



# ----
# ## Data Preparation for PCA and Clustering



# Helpful in labelling and identifying countries properly

country_data.set_index('country', inplace =True)





# #### Feature Standardization or Normalization



# Normalising continuous features

df = country_data[num_col]

country_data_norm = (df-df.mean())/df.std()

country_data_norm.head()





# ----
# ## Selection of Principal Components using PCA



#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=100)





#Doing the PCA on the train data

pca.fit(country_data_norm)





#Making the screeplot - plotting the cumulative variance against the number of components

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize = (15,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
# **Note**: Selecting 4 as number of Principal components since it explains ~ 95% variance.



# Not using Incremental PCA since our dataset is not too large and could be easily handled by PCA.

# As per official documentation,

# Incremental principal component analysis (IPCA) is typically used as a replacement for principal component analysis (PCA) when the dataset to be decomposed is too large to fit in memory.

# IPCA builds a low-rank approximation for the input data using an amount of memory which is independent of the number of input data samples.

pca_final = PCA(n_components=4,random_state=100)





# #### Basis transformation - getting the data onto our PCs



country_data_norm_pca = pd.DataFrame(pca_final.fit_transform(country_data_norm))

country_data_norm_pca.index = country_data_norm.index

country_data_norm_pca.columns = ['PC1','PC2','PC3','PC4']

country_data_norm_pca.shape





# Trying to understand the relationship between the PCs and Original columns

colnames = list(country_data_norm.columns)

pcs_df = pd.DataFrame({'PC1':pca_final.components_[0],'PC2':pca_final.components_[1],'PC3':pca_final.components_[2],'PC4':pca_final.components_[3], 'Feature':colnames})

pcs_df





# #### Creating correlation matrix for the principal components - we expect little to no correlation



#creating correlation matrix for the principal components

corrmat = np.corrcoef(country_data_norm_pca.transpose())



#plotting the correlation matrix

get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (8,8))

sns.set(style="whitegrid",font_scale=1)

sns.heatmap(corrmat, annot = True, linewidths=2, square=True, cmap="YlGnBu")





# ----
# ## Outlier Analysis and Treatment



# Removing outliers > +2 Std. Deviation or < -2 Std. Deviation 

# Checking if outliers exist

from scipy import stats

z = np.abs(stats.zscore(country_data_norm_pca))

# Outlier Records

country_data_norm_pca_outliers = country_data_norm_pca[(z > 2).any(axis=1)]

# Outlier Removed Records

country_data_norm_pca_non_outliers = country_data_norm_pca[(z < 2).all(axis=1)]

country_data_norm_pca_non_outliers.head()





# ----


# ## Visualizing Principal components



# Principal Component 1 vs. Principal Component 2

ax = sns.scatterplot(x='PC1', y='PC2', data=country_data_norm_pca_non_outliers[['PC1','PC2']])





# Principal Component 1 vs. Principal Component 3

ax = sns.scatterplot(x='PC1', y='PC3', data=country_data_norm_pca_non_outliers[['PC1','PC3']])





# Principal Component 1 vs. Principal Component 4

ax = sns.scatterplot(x='PC1', y='PC4', data=country_data_norm_pca_non_outliers[['PC1','PC4']])



# Principal Component 2 vs. Principal Component 3

ax = sns.scatterplot(x='PC2', y='PC3', data=country_data_norm_pca_non_outliers[['PC2','PC3']])





# Principal Component 2 vs. Principal Component 4

ax = sns.scatterplot(x='PC2', y='PC4', data=country_data_norm_pca_non_outliers[['PC2','PC4']])



# Principal Component 3 vs. Principal Component 4

ax = sns.scatterplot(x='PC3', y='PC4', data=country_data_norm_pca_non_outliers[['PC3','PC4']])





# ----
# ## Pre K-Means Clustering Analysis



# #### Hopkins Analysis



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



hopkins(country_data_norm_pca_non_outliers)





# **Note**: According to external reference, a value for higher than 0.75 indicates a clustering tendency at the 90% confidence level. Thus we are good to proceed with K-Means Clustering
# #### Silhouette and Elbow Analysis



# Performing Silhouette Analysis

from sklearn.metrics import silhouette_score

sse_ = []

for k in range(2, 21):

    kmeans = KMeans(n_clusters=k).fit(country_data_norm_pca_non_outliers)

    sse_.append([k, silhouette_score(country_data_norm_pca_non_outliers, kmeans.labels_)])



df = pd.DataFrame(sse_)

df.columns = ['Cluster Count', 'Silhouette Score']

sns.set(style="whitegrid",font_scale=1)

ax = sns.pointplot(x='Cluster Count', y='Silhouette Score', data=df)



# sum of squared distances (Elbow Curve Analysis)

ssd = []

for num_clusters in list(range(1,21)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(country_data_norm_pca_non_outliers)

    ssd.append([num_clusters,model_clus.inertia_])



df = pd.DataFrame(ssd)

df.columns = ['Cluster Count','sum of squared distance']

plt.figure(figsize = (16,8))

sns.set(style="whitegrid",font_scale=1)

ax = sns.pointplot(x='Cluster Count', y='sum of squared distance', data=df)





# **Note**: From the above plots, we conclude that the optimal value of K (number of clusters) could be between 5, 6 or 7.



# ## K Means with K as 5

# K-Means with K=5

model_clus5 = KMeans(n_clusters = 5, max_iter=50)

model_clus5.fit(country_data_norm_pca_non_outliers)



cluster_id = pd.DataFrame(model_clus5.labels_)

cluster_id.index = country_data_norm_pca_non_outliers.index

cluster_id.columns = ['ClusterID']

cluster_id.shape



country_data_clustered = pd.concat([country_data, cluster_id], axis=1)

country_data_clustered.head()





km_clusters_child_mort = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).child_mort.mean())

km_clusters_exports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).exports.mean())

km_clusters_health = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).health.mean())

km_clusters_imports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).imports.mean())

km_clusters_income = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).income.mean())

km_clusters_inflation = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).inflation.mean())

km_clusters_life_expec = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).life_expec.mean())

km_clusters_total_fer = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).total_fer.mean())

km_clusters_gdpp = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).gdpp.mean())





df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_child_mort, km_clusters_exports, km_clusters_health, km_clusters_imports, km_clusters_income, km_clusters_inflation, km_clusters_life_expec, km_clusters_total_fer, km_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

df



# Higher child_mort will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.child_mort)



# Lesser exports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.exports)



# Lesser health will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.health)



# Higher imports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.imports)





# Lesser income will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.income)



# Higher inflation will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.inflation)



# Lower life_expec will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.life_expec)



# Higher total_fer will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.total_fer)



# Lower gdpp will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.gdpp)



# Automatically Selecting best cluster as per our business logic 

# Normalizing

country_data_clustered_norm = (df-df.mean())/df.std()

country_data_clustered_norm.drop(columns=['ClusterID'], inplace=True)



# Adding Score column

weights = np.array([1,-1,-1,1,-1,1,-1,1,-1])

score = country_data_clustered_norm.apply(lambda a:np.dot(a,weights),axis=1)

best_cluster_id = score.argmax(axis=0)



k5_set = set(country_data_clustered[country_data_clustered['ClusterID']==best_cluster_id].index)

print(k5_set)





# ## K Means with K as 6



# K-Means with K=6

model_clus6 = KMeans(n_clusters = 6, max_iter=50)

model_clus6.fit(country_data_norm_pca_non_outliers)



cluster_id = pd.DataFrame(model_clus6.labels_)

cluster_id.index = country_data_norm_pca_non_outliers.index

cluster_id.columns = ['ClusterID']

cluster_id.shape



country_data_clustered = pd.concat([country_data, cluster_id], axis=1)

country_data_clustered.head()



km_clusters_child_mort = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).child_mort.mean())

km_clusters_exports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).exports.mean())

km_clusters_health = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).health.mean())

km_clusters_imports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).imports.mean())

km_clusters_income = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).income.mean())

km_clusters_inflation = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).inflation.mean())

km_clusters_life_expec = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).life_expec.mean())

km_clusters_total_fer = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).total_fer.mean())

km_clusters_gdpp = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).gdpp.mean())



df = pd.concat([pd.Series([0,1,2,3,4,5]), km_clusters_child_mort, km_clusters_exports, km_clusters_health, km_clusters_imports, km_clusters_income, km_clusters_inflation, km_clusters_life_expec, km_clusters_total_fer, km_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

df



# Higher child_mort will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.child_mort)





# Lesser exports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.exports)





# Lesser health will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.health)





# Higher imports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.imports)





# Lesser income will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.income)





# Higher inflation will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.inflation)





# Lower life_expec will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.life_expec)





# Higher total_fer will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.total_fer)



# Lower gdpp will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.gdpp)



# Automatically Selecting best cluster as per our business logic 

# Normalizing

country_data_clustered_norm = (df-df.mean())/df.std()

country_data_clustered_norm.drop(columns=['ClusterID'], inplace=True)



# Adding Score column

weights = np.array([1,-1,-1,1,-1,1,-1,1,-1])

score = country_data_clustered_norm.apply(lambda a:np.dot(a,weights),axis=1)

best_cluster_id = score.argmax(axis=0)



k6_set = set(country_data_clustered[country_data_clustered['ClusterID']==best_cluster_id].index)

print(k6_set)



# ## K Means with K as 7



# K-Means with K=7

model_clus7 = KMeans(n_clusters = 7, max_iter=50)

model_clus7.fit(country_data_norm_pca_non_outliers)





cluster_id = pd.DataFrame(model_clus7.labels_)

cluster_id.index = country_data_norm_pca_non_outliers.index

cluster_id.columns = ['ClusterID']

cluster_id.shape



country_data_clustered = pd.concat([country_data, cluster_id], axis=1)

country_data_clustered.head()



km_clusters_child_mort = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).child_mort.mean())

km_clusters_exports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).exports.mean())

km_clusters_health = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).health.mean())

km_clusters_imports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).imports.mean())

km_clusters_income = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).income.mean())

km_clusters_inflation = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).inflation.mean())

km_clusters_life_expec = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).life_expec.mean())

km_clusters_total_fer = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).total_fer.mean())

km_clusters_gdpp = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).gdpp.mean())





df = pd.concat([pd.Series([0,1,2,3,4,5,6]), km_clusters_child_mort, km_clusters_exports, km_clusters_health, km_clusters_imports, km_clusters_income, km_clusters_inflation, km_clusters_life_expec, km_clusters_total_fer, km_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

df



# Higher child_mort will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.child_mort)





# Lesser exports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.exports)



# Lesser health will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.health)



# Higher imports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.imports)



# Lesser income will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.income)





# Higher inflation will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.inflation)





# Lower life_expec will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.life_expec)





# Higher total_fer will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.total_fer)



# Lower gdpp will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.gdpp)





# Automatically Selecting best cluster as per our business logic 

# Normalizing

country_data_clustered_norm = (df-df.mean())/df.std()

country_data_clustered_norm.drop(columns=['ClusterID'], inplace=True)



# Adding Score column

weights = np.array([1,-1,-1,1,-1,1,-1,1,-1])

score = country_data_clustered_norm.apply(lambda a:np.dot(a,weights),axis=1)

best_cluster_id = score.argmax(axis=0)



k7_set = set(country_data_clustered[country_data_clustered['ClusterID']==best_cluster_id].index)

print(k7_set)



# Consolidated list of countries

print(k5_set | k6_set | k7_set)





# ----



# ## Hierarchical Clustering Single Method



# heirarchical clustering method 'single' color_threshold (height of cut) = 1

plt.figure(figsize = (10,30))

sns.set(style="whitegrid",font_scale=1.5)

mergings = linkage(country_data_norm_pca_non_outliers, method = "single", metric='euclidean')



dendrogram(mergings, labels=country_data.index, leaf_font_size=10, orientation='left', color_threshold=1)



from scipy import cluster

cutree = cluster.hierarchy.cut_tree(mergings, height=1)





df_cutree = pd.DataFrame(cutree)

df_cutree.index = country_data_norm_pca_non_outliers.index

df_cutree.columns = ['ClusterID']

print(df_cutree['ClusterID'].unique())

df_cutree.head()



cluster_id = df_cutree

cluster_id.shape



country_data_clustered = pd.concat([country_data, cluster_id], axis=1)

country_data_clustered.head()





km_clusters_child_mort = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).child_mort.mean())

km_clusters_exports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).exports.mean())

km_clusters_health = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).health.mean())

km_clusters_imports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).imports.mean())

km_clusters_income = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).income.mean())

km_clusters_inflation = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).inflation.mean())

km_clusters_life_expec = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).life_expec.mean())

km_clusters_total_fer = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).total_fer.mean())

km_clusters_gdpp = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).gdpp.mean())





df = pd.concat([pd.Series(df_cutree['ClusterID'].unique()), km_clusters_child_mort, km_clusters_exports, km_clusters_health, km_clusters_imports, km_clusters_income, km_clusters_inflation, km_clusters_life_expec, km_clusters_total_fer, km_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

df



# Higher child_mort will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.child_mort)



# Lesser exports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.exports)



# Lesser health will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.health)





# Higher imports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.imports)





# Lesser income will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.income)





# Higher inflation will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.inflation)



# Lower life_expec will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.life_expec)



# Higher total_fer will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.total_fer)





# Lower gdpp will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.gdpp)



# Automatically Selecting best cluster as per our business logic 

# Normalizing

country_data_clustered_norm = (df-df.mean())/df.std()

country_data_clustered_norm.drop(columns=['ClusterID'], inplace=True)



# Adding Score column

weights = np.array([1,-1,-1,1,-1,1,-1,1,-1])

score = country_data_clustered_norm.apply(lambda a:np.dot(a,weights),axis=1)

best_cluster_id = score.argmax(axis=0)



hc_s_set = set(country_data_clustered[country_data_clustered['ClusterID']==best_cluster_id].index)

print(hc_s_set)





# **Note**: We do not have a good cluster formation in this case, hence moving to next type of Hierarchichal clustering, i.e. Method Complete



# ----



# ## Hierarchical Clustering Complete Method



# heirarchical clustering method 'complete' color_threshold (height of cut) = 5

plt.figure(figsize = (10,30))

sns.set(style="whitegrid",font_scale=1.5)

mergings = linkage(country_data_norm_pca_non_outliers, method = "complete", metric='euclidean')



dendrogram(mergings, labels=country_data.index, leaf_font_size=10, orientation='left', color_threshold=5)



from scipy import cluster

cutree = cluster.hierarchy.cut_tree(mergings, height=5)



df_cutree = pd.DataFrame(cutree)

df_cutree.index = country_data_norm_pca_non_outliers.index

df_cutree.columns = ['ClusterID']

print(df_cutree['ClusterID'].unique())

df_cutree.head()



cluster_id = df_cutree

cluster_id.shape



country_data_clustered = pd.concat([country_data, cluster_id], axis=1)

country_data_clustered.head()



km_clusters_child_mort = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).child_mort.mean())

km_clusters_exports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).exports.mean())

km_clusters_health = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).health.mean())

km_clusters_imports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).imports.mean())

km_clusters_income = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).income.mean())

km_clusters_inflation = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).inflation.mean())

km_clusters_life_expec = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).life_expec.mean())

km_clusters_total_fer = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).total_fer.mean())

km_clusters_gdpp = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).gdpp.mean())





df = pd.concat([pd.Series(df_cutree['ClusterID'].unique()), km_clusters_child_mort, km_clusters_exports, km_clusters_health, km_clusters_imports, km_clusters_income, km_clusters_inflation, km_clusters_life_expec, km_clusters_total_fer, km_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

df



# Higher child_mort will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.child_mort)



# Lesser exports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.exports)



# Lesser health will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.health)



# Higher imports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.imports)



# Lesser income will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.income)



# Higher inflation will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.inflation)



# Lower life_expec will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.life_expec)



# Higher total_fer will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.total_fer)



# Lower gdpp will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.gdpp)



# Automatically Selecting best cluster as per our business logic 

# Normalizing

country_data_clustered_norm = (df-df.mean())/df.std()

country_data_clustered_norm.drop(columns=['ClusterID'], inplace=True)



# Adding Score column

weights = np.array([1,-1,-1,1,-1,1,-1,1,-1])

score = country_data_clustered_norm.apply(lambda a:np.dot(a,weights),axis=1)

best_cluster_id = score.argmax(axis=0)



hc_c_set = set(country_data_clustered[country_data_clustered['ClusterID']==best_cluster_id].index)

print(hc_c_set)





# ----



# ## Combined list of countries from K-Means and Hierarchical Clustering

print((hc_c_set)&(k5_set | k6_set | k7_set))





# ----



# ## Analysis of Outlier that were dropped



# **Note**: Since we dropped outliers (countries with extreme high/low values), we might have missed the countries in extreme need of help, thus adding them back. We do so by clustering the outliers again since it contains countries with extreme good or extreme bad situation.



country_data_norm_pca_outliers



# heirarchical clustering method 'complete' color_threshold (height of cut) = 13

plt.figure(figsize = (10,10))

sns.set(style="whitegrid",font_scale=1.5)

mergings = linkage(country_data_norm_pca_outliers, method = "complete", metric='euclidean')



dendrogram(mergings, labels=country_data.index, leaf_font_size=10, orientation='left', color_threshold=13)



from scipy import cluster

cutree = cluster.hierarchy.cut_tree(mergings, height=13)



df_cutree = pd.DataFrame(cutree)

df_cutree.index = country_data_norm_pca_outliers.index

df_cutree.columns = ['ClusterID']

print(df_cutree['ClusterID'].unique())

df_cutree.head()



cluster_id = df_cutree

cluster_id.shape



country_data_clustered = pd.concat([country_data, cluster_id], axis=1)

country_data_clustered.head()





km_clusters_child_mort = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).child_mort.mean())

km_clusters_exports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).exports.mean())

km_clusters_health = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).health.mean())

km_clusters_imports = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).imports.mean())

km_clusters_income = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).income.mean())

km_clusters_inflation = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).inflation.mean())

km_clusters_life_expec = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).life_expec.mean())

km_clusters_total_fer = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).total_fer.mean())

km_clusters_gdpp = pd.DataFrame(country_data_clustered.groupby(["ClusterID"]).gdpp.mean())





df = pd.concat([pd.Series(df_cutree['ClusterID'].unique()), km_clusters_child_mort, km_clusters_exports, km_clusters_health, km_clusters_imports, km_clusters_income, km_clusters_inflation, km_clusters_life_expec, km_clusters_total_fer, km_clusters_gdpp], axis=1)

df.columns = ["ClusterID", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

df



# Higher child_mort will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.child_mort)



# Lesser exports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.exports)



# Lesser health will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.health)



# Higher imports will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.imports)



# Lesser income will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.income)



# Higher inflation will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.inflation)



# Lower life_expec will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.life_expec)



# Higher total_fer will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.total_fer)



# Lower gdpp will relate to higher funding requirement

sns.barplot(x=df.ClusterID, y=df.gdpp)





# **Note**: Though not very clear, but Cluster 1 and Cluster 5 seems to qualify for the funding as per above plots.



# Automatically Selecting best cluster as per our business logic 

# Normalizing

country_data_clustered_norm = (df-df.mean())/df.std()

country_data_clustered_norm.drop(columns=['ClusterID'], inplace=True)



# Adding Score column

weights = np.array([1,-1,-1,1,-1,1,-1,1,-1])

score = country_data_clustered_norm.apply(lambda a:np.dot(a,weights),axis=1)

best_cluster_id = score.argmax(axis=0)



o_hc_c_set = set(country_data_clustered[country_data_clustered['ClusterID']==best_cluster_id].index)

print(o_hc_c_set)





# ----



# ## Combining, Visualizing and Summarizing Results 

# Countries which need help (worst situation)

print(o_hc_c_set)



# Countries which need help (poor situation)

print((hc_c_set)&(k5_set | k6_set | k7_set))



# Verifying the data using visual plots

df_plot = country_data_norm_pca

df_plot['country'] = df_plot.index

def needs_help(x):

    if x in o_hc_c_set:

        return 'Urgently Needed'

    elif x in ((hc_c_set)&(k5_set | k6_set | k7_set)):

        return 'Needed'

    else:

        return 'Not Needed'

df_plot['help'] = country_data_norm_pca.country.apply(needs_help)

df_plot.head()



# Plotting PC1 vs PC2 hue by country (need help or not)

sns.set(style="whitegrid",font_scale=1)

ax = sns.lmplot(x="PC1", y="PC2",data=df_plot, height=15, hue="help", fit_reg=False, markers=["o", "x", "o"])



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(df_plot.PC1, df_plot.PC2, df_plot.country, plt.gca())



# Plotting PC1 vs PC3 hue by country (need help or not)

sns.set(style="whitegrid",font_scale=1)

ax = sns.lmplot(x="PC1", y="PC3",data=df_plot, height=15, hue="help", fit_reg=False, markers=["o", "x", "o"])



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(df_plot.PC1, df_plot.PC3, df_plot.country, plt.gca())  



# Plotting PC1 vs PC4 hue by country (need help or not)

sns.set(style="whitegrid",font_scale=1)

ax = sns.lmplot(x="PC1", y="PC4",data=df_plot, height=15, hue="help", fit_reg=False, markers=["o", "x", "o"])



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(df_plot.PC1, df_plot.PC4, df_plot.country, plt.gca())  



# Plotting PC2 vs PC3 hue by country (need help or not)

sns.set(style="whitegrid",font_scale=1)

ax = sns.lmplot(x="PC2", y="PC3",data=df_plot, height=15, hue="help", fit_reg=False, markers=["o", "x", "o"])



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(df_plot.PC2, df_plot.PC3, df_plot.country, plt.gca())  



# Plotting PC2 vs PC4 hue by country (need help or not)

sns.set(style="whitegrid",font_scale=1)

ax = sns.lmplot(x="PC2", y="PC4",data=df_plot, height=15, hue="help", fit_reg=False, markers=["o", "x", "o"])



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(df_plot.PC2, df_plot.PC4, df_plot.country, plt.gca())  



# Plotting PC3 vs PC4 hue by country (need help or not)

sns.set(style="whitegrid",font_scale=1)

ax = sns.lmplot(x="PC3", y="PC4",data=df_plot, height=15, hue="help", fit_reg=False, markers=["o", "x", "o"])



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(df_plot.PC3, df_plot.PC4, df_plot.country, plt.gca())  







#**************************************************************************

#******************************* RESULT *******************************************



# Verified with plot, listing final list of countries that need funding on high priority by HELP International

print('FUNDING NEEDED AT HIGHER PRIORITY')

for k,v in enumerate(o_hc_c_set):

    output_higher_priority=(f'{k+1}. {v}')

    print(output_higher_priority)



# Verified with plot, listing final list of countries that need urgent funding on moderate priority by HELP International

print('FUNDING NEEDED AT MODERATE PRIORITY')

for k,v in enumerate((hc_c_set)&(k5_set | k6_set | k7_set)):

    output_moderate_priority=(f'{k+1}. {v}')

    print(output_moderate_priority)



print('Assignment Completed....')

# ****************** xxxxxxxxxxxxxxx ****************************