# import all libraries and dependencies for dataframe



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as plticker

%matplotlib inline



# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan



# import all libraries and dependencies for clustering

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
# Reading the country file on which analysis needs to be done



df_country = pd.read_csv('../input/pca-kmeans-hierarchical-clustering/Country-data.csv')



df_country.head()
# Reading the data dictionary file



df_structure = pd.read_csv('../input/pca-kmeans-hierarchical-clustering/data-dictionary.csv')

df_structure.head(10)
df_country.shape
df_country.describe()
df_country.info()
# Calculating the Missing Values % contribution in DF



df_null = df_country.isna().mean()*100

df_null
# Datatype check for the dataframe



df_country.dtypes
# Duplicates check



df_country.loc[df_country.duplicated()]
# Segregation of Numerical and Categorical Variables/Columns



cat_col = df_country.select_dtypes(include = ['object']).columns

num_col = df_country.select_dtypes(exclude = ['object']).columns
# Heatmap to understand the attributes dependency



plt.figure(figsize = (15,10))        

ax = sns.heatmap(df_country.corr(),annot = True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
# Pairplot of all numeric columns



sns.pairplot(df_country)
# Converting exports,imports and health spending percentages to absolute values.



df_country['exports'] = df_country['exports'] * df_country['gdpp']/100

df_country['imports'] = df_country['imports'] * df_country['gdpp']/100

df_country['health'] = df_country['health'] * df_country['gdpp']/100
df_country.head(5)
# Dropping Country field as final dataframe will only contain data columns



df_country_drop = df_country.copy()

country = df_country_drop.pop('country')
df_country_drop.head()
# Standarisation technique for scaling



warnings.filterwarnings("ignore")

scaler = StandardScaler()

df_country_scaled = scaler.fit_transform(df_country_drop)
df_country_scaled
pca = PCA(svd_solver='randomized', random_state=42)
# Lets apply PCA on the scaled data



pca.fit(df_country_scaled)
# PCA components created 



pca.components_
# Variance Ratio



pca.explained_variance_ratio_
# Variance Ratio bar plot for each PCA components.



ax = plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)

plt.xlabel("PCA Components",fontweight = 'bold')

plt.ylabel("Variance Ratio",fontweight = 'bold')
# Scree plot to visualize the Cumulative variance against the Number of components



fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.vlines(x=3, ymax=1, ymin=0, colors="r", linestyles="--")

plt.hlines(y=0.93, xmax=8, xmin=0, colors="g", linestyles="--")

plt.xlabel('Number of PCA components')

plt.ylabel('Cumulative Explained Variance')
# Checking which attributes are well explained by the pca components



org_col = list(df_country.drop(['country'],axis=1).columns)

attributes_pca = pd.DataFrame({'Attribute':org_col,'PC_1':pca.components_[0],'PC_2':pca.components_[1],'PC_3':pca.components_[2]})
attributes_pca
# Plotting the above dataframe for better visualization with PC1 and PC2



sns.pairplot(data=attributes_pca, x_vars=["PC_1"], y_vars=["PC_2"], hue = "Attribute" ,height=8)

plt.xlabel("Principal Component 1",fontweight = 'bold')

plt.ylabel("Principal Component 2",fontweight = 'bold')



for i,txt in enumerate(attributes_pca.Attribute):

    plt.annotate(txt, (attributes_pca.PC_1[i],attributes_pca.PC_2[i]))
# Plotting the above dataframe with PC1 and PC3 to understand the components which explains inflation.



sns.pairplot(data=attributes_pca, x_vars=["PC_1"], y_vars=["PC_3"], hue = "Attribute" ,height=8)

plt.xlabel("Principal Component 1",fontweight = 'bold')

plt.ylabel("Principal Component 3",fontweight = 'bold')



for i,txt in enumerate(attributes_pca.Attribute):

    plt.annotate(txt, (attributes_pca.PC_1[i],attributes_pca.PC_3[i]))
# Building the dataframe using Incremental PCA for better efficiency.



inc_pca = IncrementalPCA(n_components=3)
# Fitting the scaled df on incremental pca



df_inc_pca = inc_pca.fit_transform(df_country_scaled)

df_inc_pca
# Creating new dataframe with Principal components



df_pca = pd.DataFrame(df_inc_pca, columns=["PC_1", "PC_2","PC_3"])

df_pca_final = pd.concat([country, df_pca], axis=1)

df_pca_final.head()
# Plotting Heatmap to check is there still dependency in the dataset.



plt.figure(figsize = (8,6))        

ax = sns.heatmap(df_pca_final.corr(),annot = True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
# Scatter Plot to visualize the spread of data across PCA components



plt.figure(figsize=(20, 8))

plt.subplot(1,3,1)

sns.scatterplot(data=df_pca_final, x='PC_1', y='PC_2')

plt.subplot(1,3,2)

sns.scatterplot(data=df_pca_final, x='PC_1', y='PC_3')

plt.subplot(1,3,3)

sns.scatterplot(data=df_pca_final, x='PC_3', y='PC_2')
# Outlier Analysis 



outliers = ['PC_1','PC_2','PC_3']

plt.rcParams['figure.figsize'] = [10,8]

sns.boxplot(data = df_pca_final[outliers], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Range", fontweight = 'bold')

plt.xlabel("PC Components", fontweight = 'bold')
# Statstical Outlier treatment for PC_1



Q1 = df_pca_final.PC_1.quantile(0.05)

Q3 = df_pca_final.PC_1.quantile(0.95)

IQR = Q3 - Q1

df_pca_final = df_pca_final[(df_pca_final.PC_1 >= Q1) & (df_pca_final.PC_1 <= Q3)]



# Statstical Outlier treatment for PC_2



Q1 = df_pca_final.PC_2.quantile(0.05)

Q3 = df_pca_final.PC_2.quantile(0.95)

IQR = Q3 - Q1

df_pca_final = df_pca_final[(df_pca_final.PC_2 >= Q1) & (df_pca_final.PC_2 <= Q3)]



# Statstical Outlier treatment for PC_3



Q1 = df_pca_final.PC_3.quantile(0.05)

Q3 = df_pca_final.PC_3.quantile(0.95)

IQR = Q3 - Q1

df_pca_final = df_pca_final[(df_pca_final.PC_3 >= Q1) & (df_pca_final.PC_3 <= Q3)]
# Plot after Outlier removal 



outliers = ['PC_1','PC_2','PC_3']

plt.rcParams['figure.figsize'] = [10,8]

sns.boxplot(data = df_pca_final[outliers], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Range", fontweight = 'bold')

plt.xlabel("PC Components", fontweight = 'bold')
# Reindexing the df after outlier removal



df_pca_final = df_pca_final.reset_index(drop=True)

df_pca_final_data = df_pca_final.drop(['country'],axis=1)

df_pca_final.head()
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



#hopkins(df_pca_final_data)
# Elbow curve method to find the ideal number of clusters.

ssd = []

for num_clusters in list(range(1,8)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50,random_state= 100)

    model_clus.fit(df_pca_final_data)

    ssd.append(model_clus.inertia_)



plt.plot(ssd)
# Silhouette score analysis to find the ideal number of clusters for K-means clustering



range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state= 100)

    kmeans.fit(df_pca_final_data)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(df_pca_final_data, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

#K-means with k=4 clusters



cluster4 = KMeans(n_clusters=4, max_iter=50, random_state= 100)

cluster4.fit(df_pca_final_data)
# Cluster labels



cluster4.labels_
# Assign the label



df_pca_final['Cluster_Id4'] = cluster4.labels_

df_pca_final.head()
# Number of countries in each cluster



df_pca_final['Cluster_Id4'].value_counts()
# Scatter plot on Principal components to visualize the spread of the data



fig, axes = plt.subplots(1,2, figsize=(15,7))



sns.scatterplot(x='PC_1',y='PC_2',hue='Cluster_Id4',legend='full',palette="Set1",data=df_pca_final,ax=axes[0])

sns.scatterplot(x='PC_1',y='PC_3',hue='Cluster_Id4',legend='full',palette="Set1",data=df_pca_final,ax=axes[1])
# Lets drop the Cluster Id created with 4 clusters and proceed with 5 clusters.



df_pca_final = df_pca_final.drop('Cluster_Id4',axis=1)
#K-means with k=5 clusters



cluster5 = KMeans(n_clusters=5, max_iter=50,random_state=100)

cluster5.fit(df_pca_final_data)
# Cluster labels



cluster5.labels_
# assign the label



df_pca_final['Cluster_Id'] = cluster5.labels_

df_pca_final.head()
# Number of countries in each cluster



df_pca_final['Cluster_Id'].value_counts()
# Scatter plot on Principal components to visualize the spread of the data

fig, axes = plt.subplots(1,2, figsize=(15,7))



sns.scatterplot(x='PC_1',y='PC_2',hue='Cluster_Id',legend='full',palette="Set1",data=df_pca_final,ax=axes[0])

sns.scatterplot(x='PC_1',y='PC_3',hue='Cluster_Id',legend='full',palette="Set1",data=df_pca_final,ax=axes[1])
# Merging the df with PCA with original df



df_merge = pd.merge(df_country,df_pca_final,on='country')

df_merge_col = df_merge[['country','child_mort','exports','imports','health','income','inflation','life_expec','total_fer','gdpp','Cluster_Id']]



# Creating df with mean values

cluster_child = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).child_mort.mean())

cluster_export = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).exports.mean())

cluster_import = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).imports.mean())

cluster_health = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).health.mean())

cluster_income = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).income.mean())

cluster_inflation = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).inflation.mean())         

cluster_lifeexpec = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).life_expec.mean())

cluster_totalfer = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).total_fer.mean())

cluster_gdpp = pd.DataFrame(df_merge_col.groupby(["Cluster_Id"]).gdpp.mean())



df_concat = pd.concat([pd.Series([0,1,2,3,4]),cluster_child,cluster_export,cluster_import,cluster_health,cluster_income

                       ,cluster_inflation,cluster_lifeexpec,cluster_totalfer,cluster_gdpp], axis=1)

df_concat.columns = ["Cluster_Id", "Child_Mortality", "Exports", "Imports","Health_Spending","Income","Inflation","Life_Expectancy","Total_Fertility","GDPpcapita"]

df_concat.head()
df_merge_col.head(5)
# Scatter plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.scatterplot(x = 'income', y = 'child_mort',hue='Cluster_Id',data = df_merge_col,legend='full',palette="Set1",ax=axes[0][0])

sns.scatterplot(x = 'gdpp', y = 'income',hue='Cluster_Id', data = df_merge_col,legend='full',palette="Set1",ax=axes[0][1])

sns.scatterplot(x = 'child_mort', y = 'gdpp',hue='Cluster_Id', data=df_merge_col,legend='full',palette="Set1",ax=axes[1][0])
# Box plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.boxplot(x = 'Cluster_Id', y = 'child_mort', data = df_merge_col,ax=axes[0][0])

sns.boxplot(x = 'Cluster_Id', y = 'income', data = df_merge_col,ax=axes[0][1])

sns.boxplot(x = 'Cluster_Id', y = 'inflation', data=df_merge_col,ax=axes[1][0])

sns.boxplot(x = 'Cluster_Id', y = 'gdpp', data=df_merge_col,ax=axes[1][1])
# Box plot to visualise the mean value of few original attributes.



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.boxplot(x = 'Cluster_Id', y = 'Child_Mortality', data = df_concat,ax=axes[0][0])

sns.boxplot(x = 'Cluster_Id', y = 'Income', data = df_concat,ax=axes[0][1])

sns.boxplot(x = 'Cluster_Id', y = 'Inflation', data=df_concat,ax=axes[1][0])

sns.boxplot(x = 'Cluster_Id', y = 'GDPpcapita', data=df_concat,ax=axes[1][1])
# List of countries in Cluster 0



df_merge_col[df_merge_col['Cluster_Id']==0]
# List of countries in Cluster 3



df_merge_col[df_merge_col['Cluster_Id']==3]
df_pca_final_data.head()
# Single linkage



mergings = linkage(df_pca_final_data, method='single',metric='euclidean')

dendrogram(mergings)

plt.show()
# Complete Linkage



mergings = linkage(df_pca_final_data, method='complete',metric='euclidean')

dendrogram(mergings)

plt.show()
df_pca_hc = df_pca_final.copy()

df_pca_hc = df_pca_hc.drop('Cluster_Id',axis=1)

df_pca_hc.head()
# Let cut the tree at height of approx 3 to get 4 clusters and see if it get any better cluster formation.



clusterCut = pd.Series(cut_tree(mergings, n_clusters = 4).reshape(-1,))

df_hc = pd.concat([df_pca_hc, clusterCut], axis=1)

df_hc.columns = ['country', 'PC_1', 'PC_2','PC_3','Cluster_Id']
df_hc.head()
# Scatter plot on Principal components to visualize the spread of the data



fig, axes = plt.subplots(1,2, figsize=(15,8))



sns.scatterplot(x='PC_1',y='PC_2',hue='Cluster_Id',legend='full',palette="Set1",data=df_hc,ax=axes[0])

sns.scatterplot(x='PC_1',y='PC_3',hue='Cluster_Id',legend='full',palette="Set1",data=df_hc,ax=axes[1])
# Merging the df with PCA with original df



df_merge_hc = pd.merge(df_country,df_hc,on='country')

df_merge_col_hc = df_merge[['country','child_mort','exports','imports','health','income','inflation','life_expec','total_fer','gdpp','Cluster_Id']]
df_merge_col_hc.head()
# Scatter plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.scatterplot(x = 'income', y = 'child_mort',hue='Cluster_Id',data = df_merge_col_hc,legend='full',palette="Set1",ax=axes[0][0])

sns.scatterplot(x = 'gdpp', y = 'income',hue='Cluster_Id', data = df_merge_col_hc,legend='full',palette="Set1",ax=axes[0][1])

sns.scatterplot(x = 'child_mort', y = 'gdpp',hue='Cluster_Id', data=df_merge_col_hc,legend='full',palette="Set1",ax=axes[1][0])
df_clus0 = df_merge_col[df_merge_col['Cluster_Id'] ==0]
df_clus3 = df_merge_col[df_merge_col['Cluster_Id'] ==3]
# List of countries which need help



df_append= df_clus0.append(df_clus3)
df_append.head()
df_append.describe()
# Based on final clusters information we are going to deduce the final list.

# We observed that mean child mortality is 53 for the selected clusters and hence 

# let's take all the countries with more than this child mortality .



df_final_list = df_country[df_country['child_mort']>53]

df_final_list.shape
# Let's check the demographic of the resultant data again



df_final_list.describe()
# We observed that mean income is 3695 for the selected clusters and hence 

# let's take all the countries with less than this income .



df_final_list1 = df_final_list[df_final_list['income']<=3695]

df_final_list1.shape
# Let's check the demographic of the resultant data again



df_final_list1.describe()
# We observed that mean gdpp is 831 for the selected clusters and hence 

# let's take all the countries with less than this gdpp .



df_final_list2 = df_final_list1[df_final_list1['gdpp']<=831]

df_final_list2.shape
df_final_list2['country']
# BarPlot for Child Mortality of countries which are in need of aid



df_list_cm = pd.DataFrame(df_final_list2.groupby(['country'])['child_mort'].mean().sort_values(ascending = False))

df_list_cm.plot.bar()

plt.title('Country and Child Mortality')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("Child Mortality", fontsize = 12, fontweight = 'bold')

plt.show()
# BarPlot for Per Capita Income of countries which are in need of aid



df_list_in = pd.DataFrame(df_final_list2.groupby(['country'])['income'].mean().sort_values(ascending = False))

df_list_in.plot.bar()

plt.title('Country and Per Capita Income')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("Per Capita Income", fontsize = 12, fontweight = 'bold')

plt.show()
# BarPlot for Per Capita Income of countries which are in need of aid



df_list_gdp = pd.DataFrame(df_final_list2.groupby(['country'])['gdpp'].mean().sort_values(ascending = False))

df_list_gdp.plot.bar()

plt.title('Country and GDP per capita')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("GDP per capita", fontsize = 12, fontweight = 'bold')

plt.show()
# Final countries list

df_final_list2.reset_index(drop=True).country