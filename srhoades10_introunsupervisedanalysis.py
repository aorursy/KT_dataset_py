########################################################################

#Kaggle dataset, breast cancer proteome

#Objectives:

#Unsupervised analysis, PCA and KMeans to get a sense of variance

#Later steps would include supervised and subset selection

########################################################################
import sklearn, re

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn import preprocessing

from sklearn.decomposition import PCA
plt.style.use('seaborn-white')
#Read in the data - keep in mind last 3 columns are healthy individuals

Proteomics=pd.read_csv('../input/77_cancer_proteomes_CPTAC_itraq.csv')
IDs=pd.read_csv('../input/clinical_data_breast_cancer.csv')
#Edit some names for potential matching later

Proteomics.columns=[re.sub('\.[0-9][0-9]TCGA','',x) for x in Proteomics.columns]

IDs['Complete TCGA ID']=[re.sub('TCGA\-','',x) for x in IDs['Complete TCGA ID']]
#Code the tumor type to the patient ID

IDDict=dict(zip(IDs['Complete TCGA ID'],IDs['Tumor']))
#Add the healthy subjects

IDDict[Proteomics.columns[-3]]='Healthy'

IDDict[Proteomics.columns[-2]]='Healthy'

IDDict[Proteomics.columns[-1]]='Healthy'
#Get the X variables separate

ProteomicsXRaw=Proteomics[Proteomics.columns[3:len(Proteomics.columns)]].T
#How is the distribution of the sample intensities?

SampleIntensities=ProteomicsXRaw.sum(axis=0)

SampleDist=plt.hist(SampleIntensities.values)

plt.title('Sample Intensity Distribution')

plt.show()
#Impute missing values, scale before PCA

impute=Imputer(missing_values='NaN',strategy='mean',axis=0)

impute.fit(ProteomicsXRaw)

ProteomicsX=impute.transform(ProteomicsXRaw)
#Scaling

for inputs in range(len(ProteomicsX.T)):

    ProteomicsX.T[inputs]=preprocessing.scale(ProteomicsX.T[inputs])
#How is the distribution of the sample intensities after imputing and transforming? More suitable for PCA?

SampleIntensities=ProteomicsX.sum(axis=0)

SampleDist=plt.hist(SampleIntensities)

plt.title('Sample Intensity Distribution')

plt.show()
#PCA

pca=PCA(n_components=5)

ProteomicsX_pca=pca.fit(ProteomicsX)

ProteomicsX_pca2=ProteomicsX_pca.transform(ProteomicsX)
list(ProteomicsXRaw.index)#Plotting the first 3 components

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1,figsize=(9,6))

ax = fig.gca(projection='3d')

from collections import OrderedDict



TumorCode={'Healthy':'green','T1':'yellow','T2':'orange','T3':'red','T4':'darkviolet'}

IntermedSet=[IDDict[x] for x in list(ProteomicsXRaw.index)]

ColorSet=[TumorCode[x] for x in IntermedSet]



for i,c,ID in zip(range(len(ProteomicsX_pca2)),ColorSet,IntermedSet):

    ax.scatter3D(xs=ProteomicsX_pca2[:,0][i],

                 ys=ProteomicsX_pca2[:,1][i],

                 zs=ProteomicsX_pca2[:,2][i],

                 c=c,

                 label=ID,

                 s=90,zorder=1)



ax.set_xlabel(str.format('1st Component'+' '+str(ProteomicsX_pca.explained_variance_ratio_[0])[0:5])+'%')

ax.set_ylabel(str.format('2nd Component'+' '+str(ProteomicsX_pca.explained_variance_ratio_[1])[0:5])+'%')

ax.set_zlabel(str.format('3rd Component'+' '+str(ProteomicsX_pca.explained_variance_ratio_[2])[0:5])+'%')



plt.title('PCA of Breast Cancer Proteomics')



ColorSet, IntermedSet = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(IntermedSet, ColorSet))

plt.legend(by_label.values(), by_label.keys(),loc=3)



plt.show()
#############################

#Most variance not tumor-type

#At least not in first 3 PCs

#############################
#Plotting the first 3 components

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1,figsize=(9,6))

ax = fig.gca(projection='3d')

from collections import OrderedDict



TumorCode={'Healthy':'green','T1':'yellow','T2':'orange','T3':'red','T4':'darkviolet'}

IntermedSet=[IDDict[x] for x in list(ProteomicsXRaw.index)]

ColorSet=[TumorCode[x] for x in IntermedSet]



for i,c,ID in zip(range(len(ProteomicsX_pca2)),ColorSet,IntermedSet):

    ax.scatter3D(xs=ProteomicsX_pca2[:,0][i],

                 ys=ProteomicsX_pca2[:,1][i],

                 zs=ProteomicsX_pca2[:,2][i],

                 c=c,

                 label=ID,

                 s=90,zorder=1)



ax.set_xlabel(str.format('1st Component'+' '+str(ProteomicsX_pca.explained_variance_ratio_[0])[0:5])+'%')

ax.set_ylabel(str.format('2nd Component'+' '+str(ProteomicsX_pca.explained_variance_ratio_[1])[0:5])+'%')

ax.set_zlabel(str.format('3rd Component'+' '+str(ProteomicsX_pca.explained_variance_ratio_[2])[0:5])+'%')

ax.view_init(azim=30)



plt.title('PCA of Breast Cancer Proteomics')



ColorSet, IntermedSet = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(IntermedSet, ColorSet))

plt.legend(by_label.values(), by_label.keys(),loc=3)



plt.show()
#Clustering attemps - try 5 clusters on the PCA-reduced data

from sklearn.cluster import KMeans

clusternum=5
#Fit KMeans on the PCA Proteomics dataset

KMeansModel=KMeans(n_clusters=clusternum, init='k-means++')

KMeanData=ProteomicsX_pca2

KMeansModel.fit(KMeanData)
#Get the labels of the cluster predictions, and location of cluster centroids

labels=KMeansModel.labels_

centroids=KMeansModel.cluster_centers_
##############################################################################

#Plot the clusters and the observations with respect to the cluster boundaries

#Some of these commands are adapted from the scikit-learn example for KMeans

#... which is found at http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py (thanks!)

##############################################################################



fig = plt.figure(1,figsize=(9,6))



from collections import OrderedDict



#Further reduce to 2 components for the decision boundary plot

TwoCompReduced = PCA(n_components=2).fit_transform(ProteomicsX)

KMeansSub=KMeans(n_clusters=clusternum, init='k-means++')

KMeansSub.fit(TwoCompReduced)



# Step size - adjusted for speed here

h = .05



# Plot the decision boundary. For that, we will assign a color to each

x_min, x_max = TwoCompReduced[:, 0].min() - 1, TwoCompReduced[:, 0].max() + 1

y_min, y_max = TwoCompReduced[:, 1].min() - 1, TwoCompReduced[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# Obtain labels for each point in mesh. Use last trained model.

Z = KMeansSub.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(1)

plt.clf()

plt.imshow(Z, interpolation='nearest',

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           aspect='auto', origin='lower',alpha=0.2)



    

#Get the colors of the tumor type as used in the PCA above

TumorCode=OrderedDict([('Healthy','green'),('T1','yellow'),('T2','orange'),('T3','red'),('T4','darkviolet')])

IntermedSet=[IDDict[x] for x in list(ProteomicsXRaw.index)]

ColorSet=[TumorCode[x] for x in IntermedSet]



    

for i in range(clusternum):

    # select only data observations with cluster label == i

    DataSubset = KMeanData[np.where(labels==i)]

    

    #Get the matching list of colors by tumor type filtered by the cluster label

    MatchList=[x for x in np.where(labels==i)[0]]

    ColorList=[ColorSet[x] for x in MatchList]

    

    

    #Cluster IDs

    ClusterID=np.repeat(i,len(KMeanData[np.where(labels==i)]))

    

    for i,c,ID in zip(range(len(DataSubset)),ColorList,ClusterID):

        plt.scatter(x=DataSubset[:,0][i],

                 y=DataSubset[:,1][i],

                 c=c,

                 label=ID,s=90)

    

    

    

    #Plot positions of centroids

    plt.scatter(centroids[:, 0], centroids[:, 1],

            marker='x', s=169, linewidths=3,

            color='black', zorder=10)



 



markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='',markersize=16) for color in TumorCode.values()]

plt.legend(markers, TumorCode.keys(), numpoints=1,fontsize=16)



plt.title('K-Means of Breast Cancer Proteomics',size=20)

plt.show()
 