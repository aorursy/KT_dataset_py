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

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

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

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

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

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

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

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

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

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exports')





# Imports: Imports of goods and services. Given as %age of the Total GDP



bottom5_imports = ngo[['country','imports']].sort_values('imports', ascending = True).head()

ax = sns.barplot(x='country', y='imports', data= bottom5_imports, ax = axs[2,2])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

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
# Converting exports,imports and health spending percentages to absolute values.



ngo['exports'] = ngo['exports'] * ngo['gdpp']/100

ngo['imports'] = ngo['imports'] * ngo['gdpp']/100

ngo['health'] = ngo['health'] * ngo['gdpp']/100
ngo.head()
# Dropping Country field as final dataframe will only contain data columns



ngo_drop = ngo.copy()

country = ngo_drop.pop('country')
ngo_drop.head()
# Standarisation technique for scaling

scaler = StandardScaler()

ngo_scaled = scaler.fit_transform(ngo_drop)
ngo_scaled
pca = PCA(svd_solver='randomized', random_state=50)

# Lets apply PCA on the scaled data



pca.fit(ngo_scaled)
# PCA components created 



pca.components_
# Variance Ratio



pca.explained_variance_ratio_
# Variance Ratio bar plot for each PCA components.

plt.figure(figsize = (10, 5))

ax = plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)

plt.xlabel("PCA Components",fontweight = 'bold')

plt.ylabel("Variance Ratio",fontweight = 'bold')



plt.show()
# Scree plot to visualize the Cumulative variance against the Number of components



fig = plt.figure(figsize = (12,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.vlines(x=3, ymax=1, ymin=0, colors="r", linestyles="--")

plt.hlines(y=0.93, xmax=8, xmin=0, colors="g", linestyles="--")

plt.xlabel('Number of PCA components')

plt.ylabel('Cumulative Explained Variance')

plt.show()
# Checking which attributes are well explained by the pca components



org_col = list(ngo.drop(['country'],axis=1).columns)

attributes_pca = pd.DataFrame({'Attribute':org_col,'PC_1':pca.components_[0],'PC_2':pca.components_[1],'PC_3':pca.components_[2]})
attributes_pca
# Plotting the above dataframe for better visualization with PC1 and PC2



sns.pairplot(data=attributes_pca, x_vars=["PC_1"], y_vars=["PC_2"], hue = "Attribute" ,height=6)

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



df_inc_pca = inc_pca.fit_transform(ngo_scaled)

df_inc_pca
# Creating new dataframe with Principal components



df_pca = pd.DataFrame(df_inc_pca, columns=["PC_1", "PC_2","PC_3"])

df_pca_final = pd.concat([country, df_pca], axis=1)

df_pca_final.head()
# Plotting Heatmap to check is there still dependency in the dataset.



plt.figure(figsize = (5,5))        

ax = sns.heatmap(df_pca_final.corr(),annot = True,cmap='winter')
# Scatter Plot to visualize the spread of data across PCA components



plt.figure(figsize=(20, 5))

plt.subplot(1,3,1)

sns.scatterplot(data=df_pca_final, x='PC_1', y='PC_2')

plt.subplot(1,3,2)

sns.scatterplot(data=df_pca_final, x='PC_1', y='PC_3')

plt.subplot(1,3,3)

sns.scatterplot(data=df_pca_final, x='PC_3', y='PC_2')

plt.show()
outliers = ['PC_1','PC_2','PC_3']

plt.rcParams['figure.figsize'] = [10,5]

sns.violinplot(data = df_pca_final[outliers])

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Range", fontweight = 'bold')

plt.xlabel("PC Components", fontweight = 'bold')

plt.show()
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

plt.rcParams['figure.figsize'] = [20,5]

sns.violinplot(data = df_pca_final[outliers], orient="v", palette="Set2" )

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Range", fontweight = 'bold')

plt.xlabel("PC Components", fontweight = 'bold')

plt.show()
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

Hopkins_score=round(hopkins(df_pca_final_data),2)
print("{} is a good Hopkins score for Clustering.".format(Hopkins_score))
# Elbow curve method to find the ideal number of clusters.

ssd = []

for num_clusters in list(range(1,8)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=150,random_state= 50)

    model_clus.fit(df_pca_final_data)

    ssd.append(model_clus.inertia_)



plt.plot(ssd);
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



cluster4 = KMeans(n_clusters=4, max_iter=150, random_state= 50)

cluster4.fit(df_pca_final_data)
# Cluster labels



cluster4.labels_
# Assign the label



df_pca_final['Cluster_Id4'] = cluster4.labels_

df_pca_final.head()
# Number of countries in each cluster



df_pca_final['Cluster_Id4'].value_counts()
# Scatter plot on Principal components to visualize the spread of the data



fig, axes = plt.subplots(1,2, figsize=(15,5))



sns.scatterplot(x='PC_1',y='PC_2',hue='Cluster_Id4',legend='full',palette="Set1",data=df_pca_final,ax=axes[0])

sns.scatterplot(x='PC_1',y='PC_3',hue='Cluster_Id4',legend='full',palette="Set2",data=df_pca_final,ax=axes[1])

plt.show()
# Lets drop the Cluster Id created with 4 clusters and proceed with 5 clusters.



df_pca_final = df_pca_final.drop('Cluster_Id4',axis=1)
#K-means with k=5 clusters



cluster5 = KMeans(n_clusters=5, max_iter=120,random_state=70)

cluster5.fit(df_pca_final_data)
# Cluster labels



cluster5.labels_
# assign the label



df_pca_final['Cluster_Id'] = cluster5.labels_

df_pca_final.head()
# Number of countries in each cluster



df_pca_final['Cluster_Id'].value_counts()
# Scatter plot on Principal components to visualize the spread of the data

fig, axes = plt.subplots(1,2, figsize=(15,5))



sns.scatterplot(x='PC_1',y='PC_2',hue='Cluster_Id',legend='full',palette="Set2",data=df_pca_final,ax=axes[0])

sns.scatterplot(x='PC_1',y='PC_3',hue='Cluster_Id',legend='full',palette="Set3",data=df_pca_final,ax=axes[1])

plt.show()
# Merging the df with PCA with original df



df_merge = pd.merge(ngo,df_pca_final,on='country')

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
df_merge_col.head()
# Scatter plot on Original attributes to visualize the spread of the data



plt.figure(figsize = (20,30))

plt.subplot(3,1,1)

sns.scatterplot(x = 'income', y = 'child_mort',hue='Cluster_Id',data = df_merge_col,legend='full',palette="ocean")

plt.subplot(3,1,2)

sns.scatterplot(x = 'gdpp', y = 'income',hue='Cluster_Id', data = df_merge_col,legend='full',palette="rainbow_r")

plt.subplot(3,1,3)

sns.scatterplot(x = 'child_mort', y = 'gdpp',hue='Cluster_Id', data=df_merge_col,legend='full',palette="rainbow")

plt.show()
 #Violin plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.violinplot(x = 'Cluster_Id', y = 'child_mort', data = df_merge_col,ax=axes[0][0])

sns.violinplot(x = 'Cluster_Id', y = 'income', data = df_merge_col,ax=axes[0][1])

sns.violinplot(x = 'Cluster_Id', y = 'inflation', data=df_merge_col,ax=axes[1][0])

sns.violinplot(x = 'Cluster_Id', y = 'gdpp', data=df_merge_col,ax=axes[1][1])

plt.show()
# Violin plot to visualise the mean value of few original attributes.



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.violinplot(x = 'Cluster_Id', y = 'Child_Mortality', data = df_concat,ax=axes[0][0])

sns.violinplot(x = 'Cluster_Id', y = 'Income', data = df_concat,ax=axes[0][1])

sns.violinplot(x = 'Cluster_Id', y = 'Inflation', data=df_concat,ax=axes[1][0])

sns.violinplot(x = 'Cluster_Id', y = 'GDPpcapita', data=df_concat,ax=axes[1][1])

plt.show()
# List of countries in Cluster 0



df_merge_col[df_merge_col['Cluster_Id']==0]
# List of countries in Cluster 4



df_merge_col[df_merge_col['Cluster_Id']==4]
df_clus0 = df_merge_col[df_merge_col['Cluster_Id'] ==0]

df_clus0.head()
df_clus4 = df_merge_col[df_merge_col['Cluster_Id'] ==4]

df_clus4.head()
# List of countries which need help



df_append= df_clus0.append(df_clus4)

df_append.head()
df_append.describe()
# Based on final clusters information we are going to deduce the final list.

# We observed that mean child mortality is 54.67 for the selected clusters and hence 

# let's take all the countries with more than this child mortality .



df_final_list = ngo[ngo['child_mort']>55]

df_final_list.shape
df_final_list.head()
# Let's check the demographic of the resultant data again



df_final_list.describe()

# We observed that mean income is 3518.36 for the selected clusters and hence 

# let's take all the countries with less than this income .



df_final_list1 = df_final_list[df_final_list['income']<=3519]

df_final_list1.shape
df_final_list1.head()
df_final_list1.describe()
# We observed that mean gdpp is 831.2 for the selected clusters and hence 

# let's take all the countries with less than this gdpp .



df_final_list2 = df_final_list1[df_final_list1['gdpp']<=832]

df_final_list2.shape
df_final_list2.head()
df_final_list2.sort_values(by = ['gdpp','income','child_mort'], ascending = [True, True, False],inplace=True)

df_final_list2.head()
df_final_list2['country']
# BarPlot for Child Mortality of countries which are in need of aid



df_list_cm = pd.DataFrame(df_final_list2.groupby(['country'])['child_mort'].mean().sort_values(ascending = False))

df_list_cm.plot.bar()

plt.title('Country and Child Mortality')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("Child Mortality", fontsize = 12, fontweight = 'bold')

plt.show()
# BarPlot for Per Capita Income of countries which are in need of aid



df_list_in = pd.DataFrame(df_final_list2.groupby(['country'])['income'].mean().sort_values(ascending = True))

df_list_in.plot.bar()

plt.title('Country and Per Capita Income')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("Per Capita Income", fontsize = 12, fontweight = 'bold')

plt.show()
# BarPlot for GDP of countries which are in need of aid



df_list_gdp = pd.DataFrame(df_final_list2.groupby(['country'])['gdpp'].mean().sort_values(ascending = True))

df_list_gdp.plot.bar()

plt.title('Country and GDP per capita')

plt.xlabel("Country",fontweight = 'bold')

plt.ylabel("GDP per capita", fontsize = 12, fontweight = 'bold')

plt.show()
# Final countries list

df_final_list2.reset_index(drop=True).country[:5]