import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage

import os

print(os.listdir("../input"))

wine = pd.read_csv("../input/winequality-red.csv")

print(wine.head())

print(wine.info())

print(wine.shape)

wine.head()
f, axes = plt.subplots(3,4, figsize=(20, 12))

sns.distplot( wine["fixed acidity"], ax=axes[0,0])

sns.distplot( wine["volatile acidity"], ax=axes[0,1])

sns.distplot( wine["citric acid"], ax=axes[0,2])

sns.distplot( wine["residual sugar"], ax=axes[0,3])

sns.distplot( wine["chlorides"], ax=axes[1,0])

sns.distplot( wine["free sulfur dioxide"], ax=axes[1,1])

sns.distplot( wine["total sulfur dioxide"], ax=axes[1,2])

sns.distplot( wine["density"], ax=axes[1,3])

sns.distplot( wine["pH"], ax=axes[2,0])

sns.distplot( wine["sulphates"], ax=axes[2,1])

sns.distplot( wine["alcohol"], ax=axes[2,2])

sns.distplot( wine["quality"], ax=axes[2,3])
wine.describe()


matcorr = wine.iloc[:,~wine.columns.isin(['quality'])].corr()

mask = np.zeros_like(matcorr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 9))

cmap = sns.color_palette(sns.cubehelix_palette(8))

sns.heatmap(matcorr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True);

plt.show()
sns.lmplot(x="quality", y="fixed acidity", data=wine,

            palette="muted", height=4)



sns.lmplot(x="quality", y="volatile acidity", data=wine,

            palette="muted", height=4)



sns.lmplot(x="quality", y="citric acid", data=wine,

            palette="muted", height=4)



sns.lmplot(x="quality", y="residual sugar", data=wine,

            palette="muted", height=4)



sns.lmplot(x="quality", y="chlorides", data=wine,

            palette="muted", height=4)



sns.lmplot(x="quality", y="free sulfur dioxide", data=wine,

            palette="muted", height=4)

sns.lmplot(x="quality", y="total sulfur dioxide", data=wine,

            palette="muted", height=4)

sns.lmplot(x="quality", y="density", data=wine,

            palette="muted", height=4)

sns.lmplot(x="quality", y="alcohol", data=wine,

            palette="muted", height=4)

sns.lmplot(x="quality", y="pH", data=wine,

            palette="muted", height=4)

sns.lmplot(x="quality", y="sulphates", data=wine,

            palette="muted", height=4)
wine_single = linkage(wine.loc[:,['density','alcohol']],method="single")



plt.figure(figsize=(18,6))

dendrogram(wine_single, leaf_rotation=90)

plt.xlabel('Indicador')

plt.ylabel('Distancia')

plt.suptitle("Método Single",fontsize=18)

plt.show()
from scipy.cluster.hierarchy import fcluster

wine_SiM=wine.copy()



wine_SiM['onecluster']=fcluster(wine_single,2, criterion='maxclust')

wine_SiM['twocluster']=fcluster(wine_single,3, criterion='maxclust')

wine_SiM.head()

wine_SiM.describe()

plt.figure(figsize=(20,10))



plt.suptitle("Clustering Método Simple",fontsize=20)



plt.subplot(1,3,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiM, hue="onecluster")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiM, hue="twocluster")



plt.subplot(1,3,3)

plt.title("Quality",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiM, hue="quality")
wine_complete = linkage(wine.loc[:,['density','alcohol']],method="complete")

plt.figure(figsize=(18,6))

dendrogram(wine_complete, leaf_rotation=90)

plt.xlabel('Indicador')

plt.ylabel('Distancia')

plt.suptitle("Método Complete",fontsize=18) 

plt.show()
from scipy.cluster.hierarchy import fcluster

wine_SiC=wine.copy()



wine_SiC['oneclusterC']=fcluster(wine_complete,2, criterion='maxclust')

wine_SiC['twoclusterC']=fcluster(wine_complete,3, criterion='maxclust')

wine_SiC['threeclusterC']=fcluster(wine_complete,4, criterion='maxclust')

wine_SiC['fourclusterC']=fcluster(wine_complete,5, criterion='maxclust')



wine_SiC.head()

wine_SiC.describe()
plt.figure(figsize=(24,4))



plt.suptitle("Clustering Método Complete",fontsize=18)



plt.subplot(1,5,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiC, hue="oneclusterC")



plt.subplot(1,5,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiC, hue="twoclusterC")



plt.subplot(1,5,3)

plt.title("K = 4",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiC, hue="threeclusterC")



plt.subplot(1,5,4)

plt.title("K = 5",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiC, hue="fourclusterC")



plt.subplot(1,5,5)

plt.title("Quality",fontsize=14)

sns.scatterplot(x="density",y="alcohol", data=wine_SiC, hue="quality")