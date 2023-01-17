# Import the necessary Libraries 

import numpy as np
import pandas as pd

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Import the scaler, KMeans etc.,
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# Read the Data provided

data = pd.read_csv('Downloads//Country-data.csv')
data.head()
# Let us drop the vairable as we dont need it further for analysis.
data1 = data.drop('country',axis=1)
# Let us see the shape of the data
data1.shape
# Let us see the type of the data
data1.info()
# We could see that we dont have any non null values and the data types of ok. 
# Missing values percentage 
round(100*(data1.isnull().sum())/len(data1), 2)
# Let us see how the data is distributed.
data1.describe()
# let us convert the % age of total GDP to the actual nos. to have a better understanding of the data escpecially the exports,health and imports. 

data1['exports'] = data1['exports'] * data1['gdpp']/100
data1['imports'] = data1['imports'] * data1['gdpp']/100
data1['health'] = data1['health'] * data1['gdpp']/100
# let us see converted data now.
data1.head()
# Now we can see that we have converted the absolute data. as you may see that imports of afghanistan and albania are almost same w.r.t to gdpp thats why we have converted to the actual. 
# As also we can see the two of the datatype is int. lets convert to float as well before we proceed. 
data1['gdpp'] = data1['gdpp'].astype('float')
data1['income'] = data1['income'].astype('float')
data1.head()
# now the Data is cleaned and dont have any non null values. As we have lot of variables let us see whether there is a correlation exisit 
# Let us draw a heat map for the same to see a linear relationship is there so that we can use PCA in this case other wise we need to use the other methods.

# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(data1.corr(),annot = True)
plt.show()
## Let us see any outliers present in the numeric data sets before proceeding further.
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Health Distribution plot')
sns.distplot(data1.health)

plt.subplot(1,2,2)
plt.title('Health')
sns.boxplot(y=data1.health)

plt.show()


# Let us see the variable outliers analysis across the variables. and let us decide whether it should be treated or not.

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Gross Domestic per Capita')
sns.boxplot(y = 'gdpp', data = data1)

plt.subplot(1,2,2)
plt.title('Imports')
sns.boxplot( y = 'imports', data = data1)

plt.show()

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Exports')
sns.boxplot(y = 'exports', data = data1)

plt.subplot(1,2,2)
plt.title('Life Expectancy')
sns.boxplot(y = 'life_expec', data = data1)

plt.show()

## let us drop the country variable and start the PCA model by stadardising & Scaling first and then we proceed for the treatment of outlier analysis.

data1.head()
# Let us do the scaling of the data, the necessary scaler library has been already imported.
scaler = StandardScaler()
x = scaler.fit_transform(data1)
x
# import the necessary libraries of PCA
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized',random_state=42)
# apply the PCA
pca.fit(x)
#List of PCA components.It would be the same as the number of variables
pca.components_
#Let's check the variance ratios
pca.explained_variance_ratio_
# Import the matplot and visualse the pca variance ratio in a bar graph.
import matplotlib.pyplot as plt
plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
# most of the data in the 0 to 2 ranges.  let us see the cummulative vairance ratio.
var_cumu = np.cumsum(pca.explained_variance_ratio_)
# Make the scree plots cleary for choosing the no. of PCA 
plt.figure(figsize=(8,6))
plt.title('Scree plots')
plt.xlabel('No. of Components')
plt.ylabel('Cummulative explained variance')

plt.plot(range(1,len(var_cumu)+1), var_cumu)
plt.show()
 
#Let's try and check the first three components now
X = list(data1.columns)
newdata = pd.DataFrame({ 'Features':X,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2]})
newdata

# to make it clear let us visualise with a scatter plots of the first two principal components.
%matplotlib inline
fig = plt.figure(figsize = (10,6))
plt.scatter(newdata.PC1, newdata.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(newdata.Features):
    plt.annotate(txt, (newdata.PC1[i],newdata.PC2[i]))
plt.tight_layout()
plt.show()
# We can see that the child_mortality and total ferlities is on the PC2 and gdpp, health, income, life expectancy are in the right side of the pc1. 
#We are observing 90% variance with 3 principal components. So let's take the data until that many components
#Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=3)

#let's project our original data on the 3 principal components
df1 = pca_final.fit_transform(x)
df1.shape
#take the transpose the data, so that we can create the new matrix
pc = np.transpose(df1)
pc
#Let's create the newer matrix according to the given principal components
rownames = list(data['country'])
new_df2 = pd.DataFrame({'country':rownames,'PC1':pc[0],'PC2':pc[1],'PC3':pc[2]})
new_df2.head()
# let us do the outlier analysis now before proceeding to cluster analysis.

plt.title('Principal Component 1')
plt.boxplot(new_df2.PC1)
Q1 = new_df2.PC1.quantile(0.05)
Q3 = new_df2.PC1.quantile(0.95)
IQR = Q3- Q1
new_df2 = new_df2[(new_df2.PC1 >= Q1) & (new_df2.PC1 <=Q3)]
plt.show()
# Let us do the outlier analysis for the second principal component 2 as like 1 
plt.title('Principal Component 2')
plt.boxplot(new_df2.PC2)
Q1 = new_df2.PC2.quantile(0.05)
Q3 = new_df2.PC2.quantile(0.95)
IQR = Q3- Q1
new_df2 = new_df2[(new_df2.PC2 >= Q1) & (new_df2.PC2 <=Q3)]
plt.show()


# Let us do the outlier analysis for Principal component for 3 
plt.title('Principal Component 3')
plt.boxplot(new_df2.PC3)
Q1 = new_df2.PC3.quantile(0.05)
Q3 = new_df2.PC3.quantile(0.95)
IQR = Q3- Q1
new_df2 = new_df2[(new_df2.PC3 >= Q1) & (new_df2.PC3 <=Q3)]
plt.show()

# Let us check the data after the outlier analysis
new_df2.shape
# Let us check the new_df2 
new_df2.head()
# now let us see the scatter plot for PC1 and PC2
sns.scatterplot( x="PC1", y="PC2",data = new_df2)
# Let us use the Hopkins statistics and find the score and then evaluate it later. 
#Calculating the Hopkins statistic
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    a = X.shape[1]
    #a = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    pjd = []
    wjd = []
    for j in range(0, m):
        p_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),a).reshape(1, -1), 2, return_distance=True)
        pjd.append(p_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(pjd) / (sum(pjd) + sum(wjd))
    if isnan(H):
        print(pjd, wjd)
        H = 0
 
    return H
# #Let's check the Hopkins measure. As we need to drop the country otherwise we get a wrong representation. so country to be dropped and check the hopkins score
hopkins(new_df2.drop(['country'],axis=1))

# the Hopkins score is 80%. We can confirm it is good to perform clustering. 
# Let us remove the country coloumns which is of no use and proceed further.
new_df3 = new_df2.drop(['country'],axis=1).copy()
# check the data again;
new_df3.shape
#Let's check the silhouette score first to identify the ideal number of clusters
from sklearn.metrics import silhouette_score
sil_ = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(new_df3)
    sil_.append([k, silhouette_score(new_df3, kmeans.labels_)])
# let us plot and see what it says.
plt.plot(pd.DataFrame(sil_)[0], pd.DataFrame(sil_)[1]);
#The sihouette score reaches a peak at around 3 clusters indicating that it might be the ideal number of clusters.
#Let's use the elbow curve method to identify the ideal number of clusters.
sd = []
for no_clusters in list(range(1,10)):
    model_clus = KMeans(n_clusters = no_clusters, max_iter=100)
    model_clus.fit(new_df3)
    sd.append(model_clus.inertia_)

plt.plot(sd)
#K-means with k=5 clusters
model_clus = KMeans(n_clusters = 3, max_iter=50)
model_clus.fit(new_df3)
# let us add the cluster id to the data
data_km = pd.concat([new_df2.reset_index().drop('index', axis = 1), pd.Series(model_clus.labels_).reset_index().drop('index', axis = 1)], axis=1)
data_km.columns = ['country','PC1', 'PC2','PC3','ClusterID']
data_km.head()
# Check the count of observation per cluster
data_km['ClusterID'].value_counts()
# Plot the Cluster with respect to the clusters obtained
sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',legend='full',data=data_km)
# Plot the Cluster with respect to the clusters obtained for other PC;s.we need to make it in a better way by grouping the cluster id and compute the mean to make it better understanding.
sns.scatterplot(x='PC2',y='PC3',hue='ClusterID',legend='full',data=data_km)
# Plot the Cluster with respect to the clusters obtained for other PC;s.we need to make it in a better way by grouping the cluster id and compute the mean to make it better understanding.
sns.scatterplot(x='PC1',y='PC3',hue='ClusterID',legend='full',data=data_km)
# check the data head first and proceed further for merging in the orginial data sets.
data_km.head()
# Let's merge the original data with the data(ClusterID)
Cluster_prof=pd.merge(data,data_km, how = 'inner', on= 'country')
Cluster_prof.head()
# Let's drop PCs from the data
New_data=Cluster_prof.drop(['PC1','PC2','PC3'],axis=1)
New_data.head()
## We have performed the PCA, clusters and put back the clsuters to the original data sets now. 
# Let us do the cluster analysis for the variables provided especially the gdpp, child_mort and income variables.

clu_gdpp = pd.DataFrame(New_data.groupby(["ClusterID"]).gdpp.mean())
clu_child_mort= pd.DataFrame(New_data.groupby(["ClusterID"]).child_mort.mean())
clu_income = pd.DataFrame(New_data.groupby(["ClusterID"]).income.mean())

# Let's concat the resultant with the Cluster ID columns
CA = pd.concat([pd.Series([0,1]),clu_gdpp,clu_child_mort,clu_income], axis=1)
# Let's add column name to it
CA.columns = ["ClusterID",'gdpp','child_mort','income']
CA.head()
# Country in cluster 0 needs to be checked and retained.

Final_countryKmeans = New_data[New_data['ClusterID']==0].sort_values(by='gdpp')
Final_countryKmeans
# plots to see the three variables using boxplot 
sns.boxplot(x='ClusterID', y='income', data=Final_countryKmeans)
# plots to see the three variables using boxplot 
sns.boxplot(x='ClusterID', y='gdpp', data=Final_countryKmeans)
# plots to see the three variables using boxplot 
sns.boxplot(x='ClusterID', y='child_mort', data=Final_countryKmeans)
#let us draw a scatter plots and find the relationship between gdpp and income 
sns.scatterplot(x='gdpp',y='income',hue='ClusterID',data=New_data)
## let us proceed with the other scatter plots of gdpp with child_mort
sns.scatterplot(x='gdpp',y='child_mort',hue='ClusterID',data=New_data)
## let us proceed with the other scatter plots of income with child_mort
sns.scatterplot(x='income',y='child_mort',hue='ClusterID',data=New_data)
# let us perform the same data in herarhial clustering.
#Let's try hierarchical clustering to see if it works well
#First we'll try the single linkage procedure.
fig = plt.figure(figsize = (16,8))

Herar_clus = linkage(new_df3, method = "single", metric='euclidean')
dendrogram(Herar_clus)
plt.show()

# let us perform the same data in herarhial clustering.
#Let's try hierarchical clustering to see if it works well
#First we'll try the complete linkage procedure.
fig = plt.figure(figsize = (16,8))

Herar_clus = linkage(new_df3, method = "complete", metric='euclidean')
dendrogram(Herar_clus)
plt.show()


# now we are seeing some good clusters here. Let's see if they make sense if we eliminate the barriers
cut_cluster = pd.Series(cut_tree(Herar_clus, n_clusters = 3).reshape(-1,))
New_Hc = pd.concat([new_df2, cut_cluster], axis=1)
New_Hc.columns = ['country', 'PC1', 'PC2','PC3','ClusterID']
New_Hc.head()

# Let us merge the name with the original data with the acutal clustering what we got as like we did for k means. 
data_bk=pd.merge(data,New_Hc,on='country')
Final_bk=data_bk[['country','child_mort','exports','imports','health','income','inflation','life_expec','total_fer','gdpp','ClusterID']]
Final_bk.head()
# Check for the three variables only as like we did for the k means.
# Let us do the cluster analysis for the variables provided especially the gdpp, child_mort and income variables.

clu_gdpp = pd.DataFrame(Final_bk.groupby(["ClusterID"]).gdpp.mean())
clu_child_mort= pd.DataFrame(Final_bk.groupby(["ClusterID"]).child_mort.mean())
clu_income = pd.DataFrame(Final_bk.groupby(["ClusterID"]).income.mean())
# Let's concat the resultant with the Cluster ID columns
CA = pd.concat([pd.Series([0,1]),clu_gdpp,clu_child_mort,clu_income], axis=1)
# Let's add column name to it
CA.columns = ["ClusterID",'gdpp','child_mort','income']
CA.head()
Final_bk.drop('ClusterID',axis =1)
# 3 clusters
cluster_labels = cut_tree(Herar_clus, n_clusters=3).reshape(-1, )
cluster_labels
# assign cluster labels
Final_bk['cluster_labels'] = cluster_labels
Final_bk.head()
# Let us see the countries now with this method.
# Country in cluster 0 needs to be checked and retained.

Final_bk[Final_bk['cluster_labels']==0].sort_values(by='gdpp')

# Let us see the countries now with this method.
# Country in cluster 0 needs to be checked and retained.

Final_bk[Final_bk['cluster_labels']==1].sort_values(by='gdpp')
# Let us see the countries now with this method.
# Country in cluster 0 needs to be checked and retained.

Final_bk[Final_bk['cluster_labels']==2].sort_values(by='gdpp')
# plots to see the three variables using boxplot 
sns.scatterplot(x='income', y='child_mort', data=Final_bk)
## Infernces: seems there is no correlation exists. at the same time, as like kmeans, here the clusters are not formed properly.
# We can also see that all put together only 119 countries are there rest of 48 countires are still not been able to decide anything forwarded.
# Let us use the binning method and proceed further. 
Final_countryKmeans.sort_values(by='gdpp')
# plots to see the three variables using boxplot 
sns.boxplot(x='cluster_labels', y='gdpp', data=Final_bk)
# plots to see the three variables using boxplot 
sns.boxplot(x='cluster_labels', y='income', data=Final_bk)
# plots to see the three variables using boxplot 
sns.boxplot(x='cluster_labels', y='child_mort', data=Final_bk)
data.head()
data[data['gdpp']<=2000] 
data[data['income']<=3500]
data[data['child_mort']>=80]
# Logically these countries will be taken as the final countires to be concentrated for the funding.