# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import random 
import numpy as np 
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib as mpl
import matplotlib.pyplot as plt
#from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
Data = pd.read_excel("../input/noidea/RSU_Well Data for K Means (1).xlsx",sheet_name='Sheet1')
Data.sample(5)
print('Shape of the data set: ' + str(Data.shape))
fig, ax = plt.subplots()
n=100
colorbar = mpl.colors.ListedColormap([ 'green','cyan','yellow','magenta','blue'])
scatter=plt.scatter(Data.Vp,Data.Vs, c=Data.Facies, alpha=0.5,cmap=colorbar)
plt.title(' Vp and Vs colored by facies')
plt.xlabel('Vp (ft/s)')
plt.ylabel('Vs(ft/s)')
cbar=plt.colorbar(ticks=range(5), label='Facies')
cbar.set_ticks([1.4,2.2,3,3.8,4.6])
cbar.set_ticklabels(["Shale", "Limestone", "Sandstone", "HP Dolomite","LP Dolomite"])
plt.show()
#compute P impedance
Zp=Data.Vp*Data.RHOB
fig, ax = plt.subplots()
n=100
colorbar = mpl.colors.ListedColormap([ 'green','cyan','yellow','magenta','blue'])
scatter=plt.scatter(Data.Porosity, Zp, c=Data.Facies, alpha=0.5,cmap=colorbar)
plt.title(' Porosity and P-impedance colored by facies')
plt.xlabel('Porosity (%)')
plt.ylabel('P-impedance (ft/s*g/cc)')
cbar=plt.colorbar(ticks=range(5), label='Facies')
cbar.set_ticks([1.4,2.2,3,3.8,4.6])
cbar.set_ticklabels(["Shale", "Limestone", "Sandstone", "HP Dolomite","LP Dolomite"])
plt.show()
fig, ax = plt.subplots(figsize=(5, 15))
n=100
colorbar = mpl.colors.ListedColormap([ 'green','cyan','yellow','magenta','blue'])
scatter=plt.scatter( Data.RHOB,Data.Depth, c=Data.Facies, alpha=0.5,cmap=colorbar)
plt.title(' Density vs Depth colored by facies')
plt.xlabel('Density')
plt.ylabel('Depth')
plt.gca().invert_yaxis() # to show increasing depths
cbar=plt.colorbar(ticks=range(5), label='Facies')
cbar.set_ticks([1.4,2.2,3,3.8,4.6])
cbar.set_ticklabels(["Shale", "Limestone", "Sandstone", "HP Dolomite","LP Dolomite"])
plt.show()
Data1=Data [['Depth','RHOB','GammaRay','Vp','Vs','Porosity']]
#check the optimal k value
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data1)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()
Data1_scaled = StandardScaler().fit_transform(Data1)
#check the optimal k value
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data1_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()
km=KMeans(n_clusters=4,random_state=123,n_init=30)
km.fit(Data1_scaled)
c_labels=km.labels_
c_labels.shape
fig, ax = plt.subplots()
n=100
colorbar = mpl.colors.ListedColormap([ 'green','cyan','yellow','magenta','blue'])
#colorbar = mpl.colors.ListedColormap(['green', 'blue', 'orange', 'gray', 'pink', 'red'])
scatter=plt.scatter(Data1.Vp,Data1.Vs, c=c_labels, alpha=0.5,cmap=colorbar)
plt.title('Vp and  Vs colored with Kmeans clusters')
plt.xlabel('Vp (ft/s)')
plt.ylabel('Vs(ft/s)')
cbar=plt.colorbar(ticks=range(4), label='Cluster#')
plt.show()
fig, ax = plt.subplots()
n=100
colorbar = mpl.colors.ListedColormap([ 'yellow','green','magenta','blue'])
scatter=plt.scatter(Data1.Vp,Data1.Vs, c=c_labels, alpha=0.5,cmap=colorbar)
plt.title('Vp and  Vs colored with Kmeans clusters')
plt.xlabel('Vp (ft/s)')
plt.ylabel('Vs(ft/s)')
cbar=plt.colorbar(ticks=range(4), label='Cluster#')
plt.show()
fig, ax = plt.subplots(figsize=(5, 15))
n=100
colorbar = mpl.colors.ListedColormap([ 'yellow','green','magenta','blue'])
scatter=plt.scatter( Data.RHOB,Data.Depth, c=c_labels, alpha=0.5,cmap=colorbar)
plt.title(' Density vs Depth colored by K clusters')
plt.xlabel('Density')
plt.ylabel('Depth')
plt.gca().invert_yaxis() # to show increasing depths
cbar=plt.colorbar(ticks=range(5), label='Facies')
plt.show()
Data1_scaled.shape
pca = PCA(random_state=123) 
pca.fit(Data1_scaled)
features = range(pca.n_components_)
#check for optimal number of features
plt.figure(figsize=(8,4))
plt.bar(features[:6], pca.explained_variance_[:6], color='lightskyblue')
plt.title(' Variance covered by each component')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:6])
plt.show()
#check for optimal number of features, we plot the cumula
plt.figure(figsize=(8,4))
plt.bar(features[:6], pca.explained_variance_ratio_[:6], color='lightskyblue')
plt.title(' % Variance covered by each component')
plt.xlabel('PCA feature')
plt.ylabel('% variance')
plt.xticks(features[:6])
plt.show()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title(' Cumulative contribution of first few pca components ')
pca = PCA(n_components=3, random_state=123)
Data_pca=pca.fit_transform(Data1_scaled)
#check the optimal k value
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data_pca)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()
km_pc=KMeans(n_clusters=4,random_state=123,n_init=30)
km_pc.fit(Data_pca)
c_labels_kpc=km_pc.labels_
c_labels_kpc.shape
#------------------- Plot 1
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(15, 15))
n=100
colorbar1 = mpl.colors.ListedColormap([ 'green','cyan','yellow','magenta','blue'])
scatter1=axs[0].scatter( Data.RHOB,Data.Depth, c=Data.Facies, alpha=0.5,cmap=colorbar1)
axs[0].set_title('Original')
axs[0].set_xlabel('Density')
axs[0].set_ylabel('Depth')
#cbar=plt.colorbar(ticks=range(5), label='Facies')
cbar=fig.colorbar(scatter1,ax=axs[0],ticks=[1.4,2.2,3,3.8,4.6])
#cbar.axs[0].set_yticklabels(['Shale', 'Limestone', 'Sandstone', 'HP Dolomite','LP Dolomite'])

#------------------- Plot 2
n=100
colorbar2= mpl.colors.ListedColormap([ 'yellow','green','magenta','blue'])
#colorbar2 = mpl.colors.ListedColormap([ 'green','yellow','magenta','blue'])
scatter2=axs[1].scatter( Data.RHOB,Data.Depth, c=c_labels, alpha=0.5,cmap=colorbar2)
axs[1].set_title(' K_means before PC')
axs[1].set_xlabel('Density')
#axs[1].set_ylabel('Depth')
fig.colorbar(scatter2,ax=axs[1])


#------------------- Plot 3
n=100
#colorbar = mpl.colors.ListedColormap([ 'yellow','green','magenta','blue'])
colorbar3 = mpl.colors.ListedColormap([ 'green','yellow','magenta','blue'])
scatter3=axs[2].scatter( Data.RHOB,Data.Depth, c=c_labels_kpc, alpha=0.5,cmap=colorbar3)
axs[2].set_title(' K_means after PC')
axs[2].set_xlabel('Density')
#axs[1].set_ylabel('Depth')
fig.colorbar(scatter3,ax=axs[2])
axs[0].invert_yaxis()
axs[1].invert_yaxis()
axs[2].invert_yaxis()
plt.show()