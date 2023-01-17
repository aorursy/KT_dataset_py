# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



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
iris = pd.read_csv("../input/Iris.csv")



#Primeros registros

print(iris.head())

# No de observaciones y tipo de datos

print(iris.info())

# Numero de Observaciones y Columnas

print(iris.shape)
#Matriz de correlacion

matcorr = iris.iloc[:,~iris.columns.isin(['Id','Species'])].corr()

mask = np.zeros_like(matcorr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(matcorr, mask=mask, cmap="Purples", vmin=-1, vmax=1, center=0, square=True);

plt.show()
g = sns.PairGrid(iris.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]])

g.map_diag(plt.hist, histtype="step", linewidth=3)

g.map_offdiag(plt.scatter)
dist_sin = linkage(iris.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]],method="single")

plt.figure(figsize=(18,6))

dendrogram(dist_sin, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("DENDROGRAM SINGLE METHOD",fontsize=18)

plt.show()
from scipy.cluster.hierarchy import fcluster

iris_SM=iris.copy()



iris_SM['2_clust']=fcluster(dist_sin,2, criterion='maxclust')

iris_SM['3_clust']=fcluster(dist_sin,3, criterion='maxclust')

iris_SM.head()
plt.figure(figsize=(24,4))



plt.suptitle("Hierarchical Clustering Single Method",fontsize=18)



plt.subplot(1,3,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm", data=iris_SM, hue="2_clust")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm", data=iris_SM, hue="3_clust")



plt.subplot(1,3,3)

plt.title("Species",fontsize=14)

sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm", data=iris_SM, hue="Species")

plt.figure(figsize=(24,4))

plt.subplot(1,2,1)

plt.title("K = 2",fontsize=14)

sns.swarmplot(x="Species",y="2_clust", data=iris_SM, hue="Species")



plt.subplot(1,2,2)

plt.title("K = 3",fontsize=14)

sns.swarmplot(x="Species",y="3_clust", data=iris_SM, hue="Species")

sns.heatmap(iris_SM.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","2_clust"]].groupby(['2_clust']).mean(), cmap="Purples")
g = sns.PairGrid(iris_SM, vars=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"], hue='2_clust')

g.map(plt.scatter)

g.add_legend()
dist_comp = linkage(iris.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]],method="complete")



plt.figure(figsize=(18,6))

dendrogram(dist_comp, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("DENDROGRAM COMPLETE METHOD",fontsize=18) 

plt.show()
iris_CM=iris.copy()

iris_CM['2_clust']=fcluster(dist_comp,2, criterion='maxclust')

iris_CM['3_clust']=fcluster(dist_comp,3, criterion='maxclust')

iris_CM.head()
plt.figure(figsize=(24,4))



plt.suptitle("Hierarchical Clustering Complete Method",fontsize=18)



plt.subplot(1,3,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm", data=iris_CM, hue="2_clust")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm", data=iris_CM, hue="3_clust")



plt.subplot(1,3,3)

plt.title("Species",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm", data=iris_CM, hue="Species")
plt.figure(figsize=(24,4))

plt.subplot(1,2,1)

plt.title("K = 2",fontsize=14)

sns.swarmplot(x="Species",y="2_clust", data=iris_CM, hue="Species")



plt.subplot(1,2,2)

plt.title("K = 3",fontsize=14)

sns.swarmplot(x="Species",y="3_clust", data=iris_CM, hue="Species")

print(pd.crosstab(iris_CM["Species"],iris_CM["3_clust"]))
sns.heatmap(iris_CM.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","3_clust"]].groupby(['3_clust']).mean(), cmap="Purples")
g = sns.PairGrid(iris_CM, vars=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"], hue='3_clust')

g.map(plt.scatter)

g.add_legend()
dist_comp = linkage(iris.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]],method="ward")



plt.figure(figsize=(18,6))

dendrogram(dist_comp, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("DENDROGRAM COMPLETE METHOD",fontsize=18) 

plt.show()
iris_WM=iris.copy()

iris_WM['2_clust']=fcluster(dist_comp,2, criterion='maxclust')

iris_WM['3_clust']=fcluster(dist_comp,3, criterion='maxclust')

iris_WM['4_clust']=fcluster(dist_comp,4, criterion='maxclust')

iris_WM.head()
plt.figure(figsize=(24,4))



plt.suptitle("Hierarchical Clustering Complete Method",fontsize=18)



plt.subplot(1,4,1)

plt.title("K = 2",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="PetalWidthCm", data=iris_WM, hue="2_clust",palette="Paired")



plt.subplot(1,4,2)

plt.title("K = 3",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="PetalWidthCm", data=iris_WM, hue="3_clust",palette="Paired")



plt.subplot(1,4,3)

plt.title("K = 4",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="PetalWidthCm", data=iris_WM, hue="4_clust",palette="Paired")



plt.subplot(1,4,4)

plt.title("Species",fontsize=14)

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm", data=iris_WM, hue="Species")
plt.figure(figsize=(24,4))

plt.subplot(1,3,1)

plt.title("K = 2",fontsize=14)

sns.swarmplot(x="Species",y="2_clust", data=iris_WM, hue="Species")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=14)

sns.swarmplot(x="Species",y="3_clust", data=iris_WM, hue="Species")



plt.subplot(1,3,3)

plt.title("K = 4",fontsize=14)

sns.swarmplot(x="Species",y="4_clust", data=iris_WM, hue="Species")



print(pd.crosstab(iris_CM["Species"],iris_WM["3_clust"]))

print('_____________________________________________')

print(pd.crosstab(iris_CM["Species"],iris_WM["4_clust"]))
plt.figure(figsize=(24,4))



plt.subplot(1,2,1)

plt.title("K = 3",fontsize=14)

sns.heatmap(iris_WM.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","3_clust"]].groupby(['3_clust']).mean(), cmap="Purples")



plt.subplot(1,2,2)

plt.title("K = 4",fontsize=14)

sns.heatmap(iris_WM.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","4_clust"]].groupby(['4_clust']).mean(), cmap="Purples")
g = sns.PairGrid(iris_WM, vars=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"], hue='4_clust')

g.map(plt.scatter)

g.add_legend()