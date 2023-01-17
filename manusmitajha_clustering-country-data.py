
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


pd.set_option('display.max_rows', None )
pd.set_option('display.max_columns', None)
from sklearn.cluster import KMeans
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#loading dtaa from file
df =  pd.read_csv(r"/kaggle/input/countrydatacsv/Country-data.csv")
df.head()
#get the shape of teh data, rows and columns
df.shape
#get the numerical stats of the data
df.describe()
#get the data type of the columns
df.info()
# Calculating the Missing Values % contribution in DF

df_null = df.isna().mean()*100
df_null
# checking datatypes
df.dtypes
#creating copy of the df
df_dup = df.copy()

# Checking for duplicates and dropping the entire duplicate row if any
df_dup.drop_duplicates(subset=None, inplace=True)
#checking shape of copied dataframe after dropping duplicates
df_dup.shape
#checking shape of original dataframe
df.shape
# Converting exports,imports and health spending percentages to absolute values.

df['exports'] = df['exports'] * df['gdpp']/100
df['imports'] = df['imports'] * df['gdpp']/100
df['health'] = df['health'] * df['gdpp']/100
#inspect Df
df.head()
# Inspect
df.describe()
# Inspect df
df.head()
#we are not taking for country column as that's not used for creating density plot
features=df.columns[1:]
features
#creating this enumerated list so that we can plot the density plot of all the colummns in one go
for i in enumerate(features):
    print(i)
#plotting density plot figures for all 9 columns
plt.figure(figsize=(24,20))
for i in enumerate(features):
    plt.subplot(5,2,i[0]+1)
    sns.distplot(df[i[1]])
# Child Mortality Rate : Death of children under 5 years of age per 1000 live births
plt.figure(figsize = (30,5))
child_mort = df[['country','child_mort']].sort_values('child_mort', ascending = False)
ax = sns.barplot(x='country', y='child_mort', data= child_mort)
ax.set(xlabel = 'Country_Name', ylabel= 'Child Mortality Rate')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
child_mort_top10 = df[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
ax = sns.barplot(x='country', y='child_mort', data= child_mort_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Child Mortality Rate')
plt.xticks(rotation=90)
plt.show()
# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same
plt.figure(figsize = (30,5))
total_fer = df[['country','total_fer']].sort_values('total_fer', ascending = False)
ax = sns.barplot(x='country', y='total_fer', data= total_fer)
ax.set(xlabel = 'Country_Name', ylabel= 'Fertility Rate')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
total_fer_top10 = df[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)
ax = sns.barplot(x='country', y='total_fer', data= total_fer_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Fertility Rate')
plt.xticks(rotation=90)
plt.show()
# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same
plt.figure(figsize = (32,5))
life_expec = df[['country','life_expec']].sort_values('life_expec', ascending = True)
ax = sns.barplot(x='country', y='life_expec', data= life_expec)
ax.set(xlabel = 'Country_Name', ylabel= 'Life Expectancy')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
life_expec_bottom10 = df[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)
ax = sns.barplot(x='country', y='life_expec', data= life_expec_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Life Expectancy')
plt.xticks(rotation=90)
plt.show()
# Health :Total health spending
plt.figure(figsize = (32,5))
health = df[['country','health']].sort_values('health', ascending = True)
ax = sns.barplot(x='country', y='health', data= health)
ax.set(xlabel = 'Country_Name', ylabel= 'Health')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
health_bottom10 = df[['country','health']].sort_values('health', ascending = True).head(10)
ax = sns.barplot(x='country', y='health', data= health_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Health')
plt.xticks(rotation=90)
plt.show()
# The GDP per capita : Calculated as the Total GDP divided by the total population.
plt.figure(figsize = (32,5))
gdpp = df[['country','gdpp']].sort_values('gdpp', ascending = True)
ax = sns.barplot(x='country', y='gdpp', data= gdpp)
ax.set(xlabel = 'Country_Name', ylabel= 'GDP per capita')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
gdpp_bottom10 = df[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)
ax = sns.barplot(x='country', y='gdpp', data= gdpp_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'GDP per capita')
plt.xticks(rotation=90)
plt.show()
# Per capita Income : Net income per person
plt.figure(figsize = (32,5))
income = df[['country','income']].sort_values('income', ascending = True)
ax = sns.barplot(x='country', y='income', data=income)
ax.set(xlabel = 'Country_Name', ylabel= 'Per capita Income')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
income_bottom10 = df[['country','income']].sort_values('income', ascending = True).head(10)
ax = sns.barplot(x='country', y='income', data= income_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Per capita Income')
plt.xticks(rotation=90)
plt.show()
# Inflation: The measurement of the annual growth rate of the Total GDP
plt.figure(figsize = (32,5))
inflation = df[['country','inflation']].sort_values('inflation', ascending = False)
ax = sns.barplot(x='country', y='inflation', data= inflation)
ax.set(xlabel = 'Country_Name', ylabel= 'Inflation')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
inflation_top10 = df[['country','inflation']].sort_values('inflation', ascending = False).head(10)
ax = sns.barplot(x='country', y='inflation', data= inflation_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Inflation')
plt.xticks(rotation=90)
plt.show()
# Exports: Exports of goods and services. 
plt.figure(figsize = (32,5))
exports = df[['country','exports']].sort_values('exports', ascending = True)
ax = sns.barplot(x='country', y='exports', data= exports)
ax.set(xlabel = 'Country_Name', ylabel= 'Exports')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
exports_bottom10 = df[['country','exports']].sort_values('exports', ascending = True).head(10)
ax = sns.barplot(x='country', y='exports', data= exports_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Exports')
plt.xticks(rotation=90)
plt.show()
# Imports: Imports of goods and services. 
plt.figure(figsize = (32,5))
imports = df[['country','imports']].sort_values('imports', ascending = True)
ax = sns.barplot(x='country', y='imports', data= imports)
ax.set(xlabel = 'Country_Name', ylabel= 'Imports')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (10,5))
imports_bottom10 = df[['country','imports']].sort_values('imports', ascending = True).head(10)
ax = sns.barplot(x='country', y='imports', data= imports_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = 'Country_Name', ylabel= 'Imports')
plt.xticks(rotation=90)
plt.show()
#Bivariate Analysis
fig, axs = plt.subplots(3,3,figsize = (18,18))

# Child Mortality Rate : Death of children under 5 years of age per 1000 live births

top5_child_mort = df[['country','child_mort']].sort_values('child_mort', ascending = False).head()
ax = sns.barplot(x='country', y='child_mort', data= top5_child_mort, ax = axs[0,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Child Mortality Rate')

# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same
top5_total_fer = df[['country','total_fer']].sort_values('total_fer', ascending = False).head()
ax = sns.barplot(x='country', y='total_fer', data= top5_total_fer, ax = axs[0,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Fertility Rate')

# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same

bottom5_life_expec = df[['country','life_expec']].sort_values('life_expec', ascending = True).head()
ax = sns.barplot(x='country', y='life_expec', data= bottom5_life_expec, ax = axs[0,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Life Expectancy')

# Health :Total health spending

bottom5_health = df[['country','health']].sort_values('health', ascending = True).head()
ax = sns.barplot(x='country', y='health', data= bottom5_health, ax = axs[1,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Health')

# The GDP per capita : Calculated as the Total GDP divided by the total population.

bottom5_gdpp = df[['country','gdpp']].sort_values('gdpp', ascending = True).head()
ax = sns.barplot(x='country', y='gdpp', data= bottom5_gdpp, ax = axs[1,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'GDP per capita')

# Per capita Income : Net income per person

bottom5_income = df[['country','income']].sort_values('income', ascending = True).head()
ax = sns.barplot(x='country', y='income', data= bottom5_income, ax = axs[1,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Per capita Income')


# Inflation: The measurement of the annual growth rate of the Total GDP

top5_inflation = df[['country','inflation']].sort_values('inflation', ascending = False).head()
ax = sns.barplot(x='country', y='inflation', data= top5_inflation, ax = axs[2,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Inflation')


# Exports: Exports of goods and services.

bottom5_exports = df[['country','exports']].sort_values('exports', ascending = True).head()
ax = sns.barplot(x='country', y='exports', data= bottom5_exports, ax = axs[2,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Exports')


# Imports: Imports of goods and services

bottom5_imports = df[['country','imports']].sort_values('imports', ascending = True).head()
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
#multivariate analysis
sns.pairplot(df,diag_kind="kde")

plt.show()
df.head()
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
hopkins(df.drop('country', axis = 1))
#plotting box plot figures for all 9 columns
plt.figure(figsize=(24,20))
for i in enumerate(features):
    plt.subplot(5,2,i[0]+1)
    sns.boxplot(x= i[1], data = df)
# calculate the values of 99th percentile for exports, health, imports, income, inflation, total_fer, gdpp
q4_exports= df['exports'].quantile(.95)
q4_imports= df['imports'].quantile(.95)
q4_health= df['health'].quantile(.95)
q4_income= df['income'].quantile(.95)
q4_inflation= df['inflation'].quantile(.95)
q4_total_fer= df['total_fer'].quantile(.95)
q4_gdpp= df['gdpp'].quantile(.95)


# calculate the values of 1st percentile for life_expec
q1_life_expec= df['life_expec'].quantile(.05)
#perform Outlier capping


df['exports'][df['exports']>= q4_exports] = q4_exports

df['imports'][df['imports']>= q4_imports] = q4_imports

df['health'][df['health']>= q4_health] = q4_health

df['income'][df['income']>= q4_income] = q4_income

df['inflation'][df['inflation']>= q4_inflation] = q4_inflation

df['total_fer'][df['total_fer']>= q4_total_fer] = q4_total_fer

df['gdpp'][df['gdpp']>= q4_gdpp] = q4_gdpp



df['life_expec'][df['life_expec']<= q1_life_expec] = q1_life_expec
#check if outliers has been removed or not using box plot

#plotting box plot figures for all 9 columns
plt.figure(figsize=(24,20))
for i in enumerate(features):
    plt.subplot(5,2,i[0]+1)
    sns.boxplot(x= i[1], data = df)
# check the numerical stats after outlier removal
df.describe()
# Dropping Country field as final dataframe will only contain data columns

df_drop_country = df.copy()
country = df_drop_country.pop('country')
#inspect Df
df_drop_country.head()
# Inspect Country
country.head()
#inspect
df.describe()
#inspect df
df.info()
#scaling the dataframe df_drop_country which was created after dropping country column
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_drop_country)
df_scaled
# Inspect scaled df
df_scaled.shape
# As scaling returns an array , so we need to convert this to dataframe
df_scaled= pd.DataFrame(df_scaled, columns = df.columns[1: ])
#df_scaled= pd.DataFrame(df_scaled)
df_scaled.head()
# Inspect scaled df
df_scaled.head()
#Silhouette score , plot the graph of score for all k values from 2 to 11

from sklearn.metrics import silhouette_score
ss = []
for k in range(2, 11):
    kmean = KMeans(n_clusters = k).fit(df_scaled)
    ss.append([k, silhouette_score(df_scaled, kmean.labels_)])
temp = pd.DataFrame(ss)    
plt.plot(temp[0], temp[1])
# Plot Elbow curve for all the K values in the range 2 to 11
ssd = []
for k in range(2, 11):
    kmean = KMeans(n_clusters = k).fit(df_scaled)
    ssd.append([k, kmean.inertia_])
    
temp = pd.DataFrame(ssd)
plt.plot(temp[0], temp[1])
# K=3
# Final Kmean Clustering

kmean = KMeans(n_clusters = 3, random_state = 50) #random state is chosen so that the labes do nto change for each data point.
kmean.fit(df_scaled)
#creating a copy of DF
df_kmean = df.copy()
#labels tat we have received is an array , so converting that as an dataframe
label  = pd.DataFrame(kmean.labels_, columns= ['label'])
label.head()
#concatinating the label with the original dataframe
df_kmean = pd.concat([df_kmean, label], axis =1)
df_kmean.head()
# check what amount of data has gone to which cluster
df_kmean.label.value_counts()
# import seaborn (although its imported on top as well)
import seaborn as sns
# Plot the cluster
#taking all the numerical columns from the resulting dataframe to use them as pairplot
df_kmean_num= df_kmean[['child_mort','income', 'gdpp', 'label']]
sns.pairplot(df_kmean_num)
plt.show()

sns.jointplot(x='child_mort', y='income', data=df_kmean)
 #Violin plot to visualize the spread of the data with the labels assigned

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.violinplot(x = 'label', y = 'child_mort', data = df_kmean,ax=axes[0][0])
sns.violinplot(x = 'label', y = 'income', data = df_kmean,ax=axes[0][1])
sns.violinplot(x = 'label', y = 'inflation', data=df_kmean,ax=axes[1][0])
sns.violinplot(x = 'label', y = 'gdpp', data=df_kmean,ax=axes[1][1])
plt.show()
 # #Box plot to visualize the spread of the data with the labels assigned

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.boxplot(x = 'label', y = 'child_mort', data = df_kmean,ax=axes[0][0])
sns.boxplot(x = 'label', y = 'income', data = df_kmean,ax=axes[0][1])
sns.boxplot(x = 'label', y = 'inflation', data=df_kmean,ax=axes[1][0])
sns.boxplot(x = 'label', y = 'gdpp', data=df_kmean,ax=axes[1][1])
plt.show()
df_kmean.head()
#let's plot df with all columns exceot 'country'
df_kmean.drop('country', axis = 1).groupby('label').mean().plot(kind = 'bar')
plt.figure(figsize=(20,16))
df_kmean.drop(['country', 'exports', 'health', 'imports', 'inflation', 'life_expec','total_fer'], axis = 1).groupby('label').mean().plot(kind = 'bar')
df_kmean[df_kmean['label'] == 1].sort_values(by = ['child_mort', 'income', 'gdpp'], ascending = [False, True, True]).head(5)
#Let's also try with k=4 to see how are the clusters formed
# K=4
# Final Kmean Clustering

kmean = KMeans(n_clusters = 4, random_state = 50) #random state is chosen so that the labes do nto change for each data point.
kmean.fit(df_scaled)
#creating a copy of DF
df_kmean = df.copy()
df_kmean.head()
#labels tat we have received is an array , so converting that as an dataframe
label  = pd.DataFrame(kmean.labels_, columns= ['label'])
label.head()
#concatinating the label with the original dataframe
df_kmean = pd.concat([df_kmean, label], axis =1)
df_kmean.head()
# check what amount of data has gone to which cluster
df_kmean.label.value_counts()
# Plot the cluster
#taking all the numerical columns from the resulting dataframe to use them as pairplot
df_kmean_num= df_kmean[['child_mort','income', 'gdpp', 'label']]
sns.pairplot(df_kmean_num)
plt.show()
sns.jointplot(x='child_mort', y='income', data=df_kmean)
 #Violin plot to visualize the spread of the data with the labels assigned

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.violinplot(x = 'label', y = 'child_mort', data = df_kmean,ax=axes[0][0])
sns.violinplot(x = 'label', y = 'income', data = df_kmean,ax=axes[0][1])
sns.violinplot(x = 'label', y = 'inflation', data=df_kmean,ax=axes[1][0])
sns.violinplot(x = 'label', y = 'gdpp', data=df_kmean,ax=axes[1][1])
plt.show()
 # #Box plot to visualize the spread of the data with the labels assigned

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.boxplot(x = 'label', y = 'child_mort', data = df_kmean,ax=axes[0][0])
sns.boxplot(x = 'label', y = 'income', data = df_kmean,ax=axes[0][1])
sns.boxplot(x = 'label', y = 'inflation', data=df_kmean,ax=axes[1][0])
sns.boxplot(x = 'label', y = 'gdpp', data=df_kmean,ax=axes[1][1])
plt.show()
plt.figure(figsize=(24,24))
df_kmean.drop(['country', 'exports', 'health', 'imports', 'inflation', 'life_expec','total_fer'], axis = 1).groupby('label').mean().plot(kind = 'bar')
# Inspect
df_scaled.head()
#importing libraries for Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# Single linkage
plt.figure(figsize=(24,20))
mergings = linkage(df_scaled, method='single',metric='euclidean')
dendrogram(mergings)
plt.show()
plt.figure(figsize=(24,20))
mergings = linkage(df_scaled, method='complete',metric='euclidean')
dendrogram(mergings)
plt.show()
#now let's cut this dendogram at appropriate level for 4 clusters
cut_tree(mergings, n_clusters=4)
#reshaping cluster labels to dataframe
cluster_labels=cut_tree(mergings, n_clusters=4).reshape(-1, )
#inspect the scaled df
df_scaled.head()
#ispect original df
df.head()
#assign the cluster labels
df['cluster_labels']=cluster_labels
#ispect original df after adding original df
df.head()
 #Box plot on Original attributes to visualize the spread of the data

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.boxplot(x = 'cluster_labels', y = 'child_mort', data = df,ax=axes[0][0])
sns.boxplot(x = 'cluster_labels', y = 'income', data = df,ax=axes[0][1])
sns.boxplot(x = 'cluster_labels', y = 'inflation', data=df,ax=axes[1][0])
sns.boxplot(x = 'cluster_labels', y = 'gdpp', data=df,ax=axes[1][1])
plt.show()
 #VoilinPlot plot on Original attributes to visualize the spread of the data

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.violinplot(x = 'cluster_labels', y = 'child_mort', data = df,ax=axes[0][0])
sns.violinplot(x = 'cluster_labels', y = 'income', data = df,ax=axes[0][1])
sns.violinplot(x = 'cluster_labels', y = 'inflation', data=df,ax=axes[1][0])
sns.violinplot(x = 'cluster_labels', y = 'gdpp', data=df,ax=axes[1][1])
plt.show()
# Inspect df
df.head()
df.drop(['country', 'exports', 'health', 'imports', 'inflation', 'life_expec','total_fer'], axis = 1).groupby('cluster_labels').mean().plot(kind = 'bar')
#complete linkage, cut tree for cluster =3
#now let's cut this dendogram at appropriate level for 3 clusters
cut_tree(mergings, n_clusters=3)
# reshape to df
cluster_labels=cut_tree(mergings, n_clusters=3).reshape(-1, )
# inspect scaled df
df_scaled.head()
#Inspect original df
df.head()
#assign the cluster labels with a different column name this time cluster_labels2 as it already has the previous labels from 4 clusters
df['cluster_labels2']=cluster_labels
df.head()
 #Box plot on Original attributes to visualize the spread of the data

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.boxplot(x = 'cluster_labels2', y = 'child_mort', data = df,ax=axes[0][0])
sns.boxplot(x = 'cluster_labels2', y = 'income', data = df,ax=axes[0][1])
sns.boxplot(x = 'cluster_labels2', y = 'inflation', data=df,ax=axes[1][0])
sns.boxplot(x = 'cluster_labels2', y = 'gdpp', data=df,ax=axes[1][1])
plt.show()
 #VoilinPlot plot on Original attributes to visualize the spread of the data

fig, axes = plt.subplots(2,2, figsize=(15,12))

sns.violinplot(x = 'cluster_labels2', y = 'child_mort', data = df,ax=axes[0][0])
sns.violinplot(x = 'cluster_labels2', y = 'income', data = df,ax=axes[0][1])
sns.violinplot(x = 'cluster_labels2', y = 'inflation', data=df,ax=axes[1][0])
sns.violinplot(x = 'cluster_labels2', y = 'gdpp', data=df,ax=axes[1][1])
plt.show()
#Inspect df
df.head()
df.drop(['country', 'exports', 'health', 'imports', 'inflation', 'life_expec','total_fer', 'cluster_labels'], axis = 1).groupby('cluster_labels2').mean().plot(kind = 'bar')
df[df['cluster_labels2'] == 0].sort_values(by = ['child_mort', 'income', 'gdpp'], ascending = [False, True, True]).head(5)
