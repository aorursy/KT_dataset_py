import  numpy as np

import  pandas as pd

import seaborn as sn

import missingno as msno

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

from sklearn.decomposition import PCA 
import sklearn.metrics as sm

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import cophenet

from scipy.spatial.distance import pdist
data = pd.read_csv("../input/wholesale_customers.csv")

data.drop(['Region','Channel'], axis = 1, inplace = True)
#Display the data

data.describe()
#Missing data detection

msno.matrix(data,figsize=(10,3))

#Data distribution

sn.boxplot(data=data, orient="v")



#We can see that the last feature is pretty low.

#We also see a lot of outliers and you may want to remove them from the dataset.
#Let me see  the correlation of the data and maybe we can see that we need to be more one of these because they are carrying high

#correlation between 

#But this groceries actually the feature that we kind of want to predict.

#So, in essence, we will remove this grocery corn from our dataset while building model.

corrMatt=data.corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)]=False

fig,ax=plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True,annot=True)



#Scatterplot

mx_plot = sn.pairplot(data, diag_kind="kde", size=1.6)

mx_plot.set(xticklabels=[])



#So you can see that correlation here and you can also see that grocery and milk are as well  having some kind of 

#correlation between that and that is why they're going to remove from the dataset.
##Implementing hierarchical clustering 

#We are going to drop the grocery because we detected



X = data.drop(["Grocery"], axis = 1)
# Scale data

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Create dendragram

#Decide how many cluster we need ?





plt.figure(figsize=(20, 10))  

dendagram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title("Dendogram")





plt.show()
#This is the longest line that we have for crossing and the 



plt.figure(figsize=(20, 10))  

dendagram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title("Dendogram")





plt.axhline(y=20)

plt.show()
# Creating 

## If linkage is “ward”, only “euclidean” is accepted.

ward_euclidean = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_out_ward = ward_euclidean.fit_predict(X)

# generate the linkage matrix

Z_ward = linkage(X, 'ward')
c, coph_dists = cophenet(Z_ward, pdist(X))

c

pca_2 = PCA(2) # Two Canonical Variables

plot_columns = pca_2.fit_transform(data)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=ward_euclidean.labels_)

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 5 Clusters with "ward and euclidean" ')

plt.show()


complete_euclidean = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'complete')

y_out_complete_euclidean = complete_euclidean.fit_predict(X)

# generate the complete matrix

Z_complete = linkage(X, 'complete')
c, coph_dists = cophenet(Z_complete, pdist(X))

c
pca_2 = PCA(2) # Two Canonical Variables

plot_columns = pca_2.fit_transform(data)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=complete_euclidean.labels_)

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 5 Clusters with "complete and euclidean"')

plt.show()
complete_manhattan = AgglomerativeClustering(n_clusters = 5, affinity = 'manhattan', linkage = 'complete')

y_out_complete_manhattan = complete_manhattan.fit_predict(X)
pca_2 = PCA(2) # Two Canonical Variables

plot_columns = pca_2.fit_transform(data)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=complete_manhattan.labels_)

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 5 Clusters with "complete and manhattan"')

plt.show()
complete_l2 = AgglomerativeClustering(n_clusters = 5, affinity = 'l2', linkage = 'complete')

y_out_complete_l2 = complete_l2.fit_predict(X)
pca_2 = PCA(2) # Two Canonical Variables

plot_columns = pca_2.fit_transform(data)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=complete_l2.labels_)

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 5 Clusters with "complete and l2"')

plt.show()
average_euclidean = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',  linkage = 'average')

y_out_average_euclidean = average_euclidean.fit_predict(X)
# generate the average matrix

Z_average = linkage(X, 'average')
c, coph_dists = cophenet(Z_average, pdist(X))

c
pca_2 = PCA(2) # Two Canonical Variables

plot_columns = pca_2.fit_transform(data)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=average_euclidean.labels_)

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 5 Clusters with "average and euclidean"')

plt.show()
average_manhattan = AgglomerativeClustering(n_clusters = 5, affinity = 'manhattan', linkage = 'average')

y_out_average_manhattan = average_manhattan.fit_predict(X)
pca_2 = PCA(2) # Two Canonical Variables

plot_columns = pca_2.fit_transform(data)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=average_manhattan.labels_)

plt.xlabel('Canonical variable 1')

plt.ylabel('Canonical variable 2')

plt.title('Scatterplot of Canonical Variables for 5 Clusters with "average and manhattan"')

plt.show()