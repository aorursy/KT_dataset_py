import scipy.cluster.hierarchy as hca

import pandas as pd

import numpy as np

from pylab import *

from matplotlib import pyplot

%matplotlib inline

np.set_printoptions(suppress=True)

from sklearn import cluster

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs

from scipy.cluster.hierarchy import cophenet

from scipy.spatial.distance import pdist



pantheon_data=pd.read_csv("../input/database.csv")

pantheon_data.head(5)



pantheon_data.tail(5)
pantheon_data.describe()
pantheon_data.dtypes
pantheon_data.count()



#notice there are several columns with NaNs since they are less than 11341
pantheon_data.shape
#Dropping "article_id(column 0), full_name (colulmn 1) & "state" (column 5) & city (column 4) since have Nans that don't make

#sense to replace or with the case of country, add too much complexity to the dataset



p_data = pantheon_data.drop(pantheon_data.columns[[0, 1, 4, 5]], axis=1)
p_data.shape
p_data.head()
# sorting dataset by country column to see which rows have NaN data



p_data_sort_country = p_data.sort_values(by = ['country'], ascending = True)

p_data_sort_country.tail(34)



# Missing Countries by looking at cities



#Victoria Peak= Hong Kong

#Lisala =  Congo

#Yangon = Myanmar

#Tórshavn = Faroe Islands

#Cetinje = Montenegro

#Saint-Denis = France

#Kyaukse = Myanmar

#Manatuto = EastTimor

#Paungdale = Myanmar

#Bongoville = Gabon



# Natmauk, Myanmar

# Yangon, Myanmar

# Dili, East Timor

# Ngapudaw Township, Myanmar

# Kowloon, Hong Kong

# Libreville, Gabon

# Mouila, Gabon

# Fizi Territory, Congo

# Nikšić, Montenegro

# Mbandaka, Congo

# Kinshasa, Congo

# Podgorica, Montenegro



# Myanmar = Yangon,Kyauks, Natmauk

# Hong Kong = Victoria Peak, Kowloon

# Congo = Lisala, Fizi Territory, Mbandaka, Kinshasa, Podgorica 

# Faroe Islands = Tórshavn

# Montenegro = Cetinje, Nikšić

# France = Saint-Denis

# East Timor = Manatuto, Dili

# Gabon = Bongoville, Libreville, Mouila





#sorting dataset by continent to see which rows have NaNs



p_data_sort_continent = p_data.sort_values(by = ['continent'], ascending = True)

p_data_sort_continent.tail(31)

#Filling in NaNs with "Unknown" in the Country and Continent columns



p_data["country"].fillna(value="Unknown", inplace=True)

p_data["continent"].fillna(value="Unknown", inplace = True)

p_data.head()



#filling in NaNs in the latitude and longitude columns with 0



p_data["latitude"].fillna(value=0, inplace=True)

p_data["longitude"].fillna(value=0, inplace=True)

p_data.head()

p_data.dtypes
#checking to see all of the NaNs were filled in.  All row counts should be 11341.



p_data.count()
# Found out that HDI column had some rows without data



p_data_sort_HDI = p_data.sort_values(by = ['historical_popularity_index'], ascending = True)

p_data_sort_HDI.tail(6)
# Filled in the rows without data in HDI column with 0



p_data_sort_HDI = p_data_sort_HDI.replace(['Not Provided'], 0)

p_data_sort_HDI.tail(6)
# In case other columns have cells with "not provided", replacing all cells with 'not provided' with 0



p_data = p_data.replace(['Not Provided'], 0)

p_data.tail(6)
p_data.dtypes
# changing the columns that look like numbers but show up as object to numeric



p_data[["article_languages", "page_views","average_views", "historical_popularity_index"]] = p_data[["article_languages", "page_views","average_views", "historical_popularity_index"]].apply(pd.to_numeric)

p_data.head()



# saw that row birth year for row 1522 was unknown



p_data.ix[1522]
# replacing birth year which was unknown to be 0

p_data['birth_year'] = p_data['birth_year'].replace(['Unknown'], 0)
p_data.ix[1522]
p_data.tail()
# saw that birth year wasn't a real year



p_data.ix[3009]
# replaced birth year with an actual year



p_data['birth_year'] = p_data['birth_year'].replace(['1237?'], 1237)
# saw birth year was a guess and changed it to a real year



p_data['birth_year'] = p_data['birth_year'].replace(['530s'], 530)
# made birth year results numeric



p_data[["birth_year"]] = p_data[["birth_year"]].apply(pd.to_numeric)
p_data.dtypes
p_data.head()
# creating dummy variables for the columns that were objects



p_data_dummies = pd.get_dummies(p_data[['sex','country','continent','occupation','industry','domain']])

p_data1 = pd.concat([p_data, p_data_dummies], axis=1)

p_data1.head()
#Dropping original columns converted to dummy variarables 



p_data2 = p_data1.drop(p_data1.columns[[0,2,3,6,7,8]], axis=1)

p_data2.head()
#normalizing the data



stscaler = StandardScaler().fit(p_data2)

p_data2Norm = stscaler.transform(p_data2)

p_data2Norm
####### starting hierarcharial clustering #######
link_matrix=hca.linkage(p_data2Norm,metric="euclidean",method="ward")
plt=hca.dendrogram(link_matrix,truncate_mode="lastp",p=25)

xticks(rotation=90)

ylabel("Distance")

figtext(0.5,0.95,"Normalized Pantheon Data",ha="center",fontsize=12)

figtext(0.5,0.90,"Dendrogram (center, euclidean, ward)",ha="center",fontsize=10)
fig, axes = pyplot.subplots(1, 1, figsize=(10, 10))

plt=hca.dendrogram(link_matrix,truncate_mode="lastp",p=25)

xticks(rotation=90)

ylabel("Distance")

figtext(0.5,0.95,"Normalized Airline Data",ha="center",fontsize=12)

figtext(0.5,0.90,"Dendrogram (center, euclidean, ward)",ha="center",fontsize=10)

axhline(y=235)
from scipy.cluster.hierarchy import fcluster





k=7

clusters = fcluster(link_matrix, k, criterion='maxclust')

clusters
pd.value_counts(pd.Series(clusters))
df_clusters=pd.DataFrame(clusters)

df_clusters.head()
p_data2['cluster']=df_clusters

p_data2.head()
grouped_df_clusters = p_data2.sort_values(by = ['cluster'], ascending = True)

grouped_df_clusters.head(10)
np.set_printoptions(suppress=True)



p_data2_count = p_data2.groupby(['cluster']).count().round(0)

p_data2_count
# Supress scientific notation. Output numbers look neat without 10 digit long precision values

np.set_printoptions(suppress=True)



p_data2_mean = p_data2.groupby(['cluster']).mean()

p_data2_mean
#Dropping hierarchacle cluster column

p_data3 = p_data2.drop(p_data2.columns[[336]], axis=1)

p_data3.head()

p_data3.shape
########## starting k-means clustering  #########



p_data3_array = np.array(p_data3)



# Print first five rows of data

p_data3_array[0:5]
k = 3

kmeans = cluster.KMeans(n_clusters=k)

kmeans.fit(p_data3_array)
labels = kmeans.labels_

centroids = kmeans.cluster_centers_
centroids
for i in range(k): # repeat loop n times for n clusters

    ds = p_data3_array[np.where(labels==i)] # Get the data for indexes where label is equal to a particular cluster

    pyplot.plot(ds[:,0],ds[:,1],'o')  

    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')

    pyplot.setp(lines,ms=15.0)

    pyplot.setp(lines,mew=2.0)

pyplot.show()
pd.value_counts(pd.Series(labels))
df2_clusters=pd.DataFrame(labels)

df2_clusters.head()
p_data2['k-means cluster']=df2_clusters

p_data2.head()
grouped_df_clusters2 = p_data2.sort_values(by = ['k-means cluster'], ascending = True)

grouped_df_clusters2.head(10)
np.set_printoptions(suppress=True)



p_data2km_count = p_data2.groupby(['k-means cluster']).count().round(0)

p_data2km_count
# Supress scientific notation. Output numbers look neat without 10 digit long precision values

np.set_printoptions(suppress=True)



p_data2km_mean = p_data2.groupby(['k-means cluster']).mean()

p_data2km_mean
######## steps for PCA from article #######



#pca = PCA(n_components=44)



#pca.fit(X)



#The amount of variance that each PC explains

#var= pca.explained_variance_ratio_



#Cumulative Variance explains

#var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)



#print var1

#[  10.37   17.68   23.92   29.7    34.7    39.28   43.67   46.53   49.27

#51.92   54.48   57.04   59.59   62.1    64.59   67.08   69.55   72.

#74.39   76.76   79.1    81.44   83.77   86.06   88.33   90.59   92.7

#94.76   96.78   98.44  100.01  100.01  100.01  100.01  100.01  100.01

#100.01  100.01  100.01  100.01  100.01  100.01  100.01  100.01]



#plt.plot(var1)



#Looking at above plot I'm taking 30 variables

#pca = PCA(n_components=30)

#pca.fit(X)

#X1=pca.fit_transform(X)



#print X1
########### PCA Analysis ##########
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt



from sklearn.cluster import DBSCAN

from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs

from sklearn.preprocessing import StandardScaler
# Running PCA analysis with 2 components



pca = PCA(n_components=2)
# Fitting the standardized data



pca.fit(p_data2Norm)
# Transform the standardized data into an array



pan_pca = pca.transform(p_data2Norm)

pan_pca
# Create a dataframe



pan_pca_df = pd.DataFrame(pan_pca)
pan_pca_df.index = p_data2.index


pan_pca_df.head()
# Added labels to dataframe



pan_pca_df.columns = ['PC1','PC2']

pan_pca_df.head()
####### DBSCAN Analysis ########
data = np.array(pan_pca_df)

data
#Run the DBSCAN algorithm on Data to construct a DBSCAN object



db = DBSCAN(eps = 0.5, min_samples = 100).fit(pan_pca_df)

db
print(db.labels_)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# Assign "True" to the indexes in core_samples_mask list for the samples where dbscan was able to cluster the points.



core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

labels
# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt



from sklearn.cluster import DBSCAN

from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs

from sklearn.preprocessing import StandardScaler







unique_labels = set(labels)

colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

colors
indexes= zip(unique_labels, colors)

list(indexes)
for(label, color) in zip(unique_labels, colors):

    class_member_mask = (labels == label)

    xy = data[class_member_mask & core_samples_mask]

    plt.plot(xy[:,0],xy[:,1], 'o', markerfacecolor = color, markersize = 10)

    xy2 = data[class_member_mask & ~core_samples_mask]

    plt.plot(xy2[:,0],xy2[:,1], 'o', markerfacecolor = color, markersize = 5)

plt.title("DBSCAN on Pantheon data")

plt.xlabel("article_languages")

plt.ylabel("historical_popularity_index")

#Print the outliers which are the labels that couldn't be assigned to any cluster

outliers_cluster = pantheon_data.loc[labels==-1,]

outliers_cluster
cluster1 = pantheon_data.loc[labels==1,]

cluster1
cluster0 = pantheon_data.loc[labels==0,]

cluster0
outliers_cluster_men = outliers_cluster[outliers_cluster["sex"]=="Male"]

outliers_cluster_men.sort_values(by = ['domain', 'industry','occupation'])

outliers_cluster_men_sports = outliers_cluster_men[outliers_cluster_men["domain"]=="Sports"]

outliers_cluster_men_sports.sort_values(by = ['domain', 'industry','occupation'])
outliers_cluster_men_other = outliers_cluster_men[outliers_cluster_men["domain"]!="Sports"]

outliers_cluster_men_other.sort_values(by = ['domain', 'industry','occupation'])
outliers_cluster_men_other_Institutions = outliers_cluster_men_other[outliers_cluster_men_other["domain"]=="Institutions"]

outliers_cluster_men_other_Institutions.sort_values(by = ['domain', 'industry','occupation'])
outliers_cluster_men_other_Institutions_US = outliers_cluster_men_other_Institutions[outliers_cluster_men_other_Institutions["country"]=="United States"]

outliers_cluster_men_other_Institutions_US.sort_values(by = ['domain', 'industry','occupation'])
cluster0_institutions = cluster0[cluster0['domain']=='Institutions']

cluster0_institutions.sort_values(by = ['domain', 'industry','occupation'])
cluster0_institutions_US = cluster0_institutions[cluster0_institutions['country']=='United States']

                                        

cluster0_institutions_US.sort_values(by = ['domain', 'industry','occupation'])
cluster0_institutions_US_politician = cluster0_institutions_US[cluster0_institutions_US['occupation']=='Politician']

                                        

cluster0_institutions_US_politician.sort_values(by = ['historical_popularity_index'])
outliers_cluster_men_other_humanities = outliers_cluster_men_other[outliers_cluster_men_other["domain"]=="Humanities"]

outliers_cluster_men_other_humanities.sort_values(by = ['domain', 'industry','occupation'])
outlier_mean = p_data2.loc[labels==-1,].mean().round(2)

outlier_mean
cluster1_mean = p_data2.loc[labels==1,].mean().round(2)

cluster1_mean
cluster0_mean = p_data2.loc[labels==0,].mean().round(2)

cluster0_mean