# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importa Packages 

import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Upload data

movies = pd.read_csv('../input/movies.csv', encoding = "ISO-8859-1")



#Visualización de los 5 primeros registros

print(movies.head())

# informacion de las variables

print(movies.info())

# shape de los datos

print(movies.shape)

# Descripcion de los datos

movies.describe()
# Compute the correlation matrix

corr=movies.iloc[:,~movies.columns.isin(['company','country','director','genre','name','released',

                                         'star','writer'])].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()



g = sns.PairGrid(movies.loc[:,["budget","gross","runtime","score","votes","year"]])

g.map_diag(plt.hist, histtype="step", linewidth=3)

g.map_offdiag(plt.scatter)
sin = linkage(movies.loc[:,["gross","score"]],method="single")

#"budget","gross","runtime","score","votes","year"

plt.figure(figsize=(18,6))

dendrogram(sin, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Euclidean distances')

plt.suptitle("DENDROGRAM SINGLE METHOD",fontsize=8)

plt.show()

from scipy.cluster.hierarchy import fcluster

movies_clus=movies.copy()



movies_clus['2_clust']=fcluster(sin,2, criterion='maxclust')

movies_clus['3_clust']=fcluster(sin,3, criterion='maxclust')

movies_clus['10_clust']=fcluster(sin,10, criterion='maxclust')

movies_clus.head()



plt.figure(figsize=(24,4))



plt.suptitle("Hierarchical Clustering Single Method",fontsize=18)



plt.subplot(1,3,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="gross",y="votes", data=movies_clus, hue="2_clust")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="gross",y="votes", data=movies_clus, hue="3_clust")



plt.subplot(1,3,3)

plt.title("K = 10",fontsize=14)

sns.scatterplot(x="gross",y="votes", data=movies_clus, hue="10_clust")

g = sns.PairGrid(movies_clus, vars=["budget","gross","runtime","score","votes","year"], hue='3_clust',palette="RdBu")

g.map(plt.scatter)

g.add_legend()

com = linkage(movies.loc[:,["gross","score"]],method="complete")

#"budget","gross","runtime","score","votes","year"

plt.figure(figsize=(18,6))

dendrogram(com, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Euclidean distances')

plt.suptitle("DENDROGRAM COMPLETE METHOD",fontsize=8)

plt.show()

from scipy.cluster.hierarchy import fcluster

movies_clus=movies.copy()



movies_clus['2_clustCom']=fcluster(com,2, criterion='maxclust')

movies_clus['3_clustCom']=fcluster(com,3, criterion='maxclust')

movies_clus['10_clustCom']=fcluster(com,10, criterion='maxclust')

movies_clus.head()



plt.figure(figsize=(24,4))



plt.suptitle("Hierarchical Clustering Complete Method",fontsize=18)



plt.subplot(1,3,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="gross",y="votes", data=movies_clus, hue="2_clustCom")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="gross",y="votes", data=movies_clus, hue="3_clustCom")



plt.subplot(1,3,3)

plt.title("K = 10",fontsize=14)

sns.scatterplot(x="gross",y="votes", data=movies_clus, hue="10_clustCom")

g = sns.PairGrid(movies_clus, vars=["budget","gross","score","votes"], hue='3_clustCom', palette="RdBu")

g.map(plt.scatter)

g.add_legend()