import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
wine = pd.read_csv("../input/Wine.csv")



#Primeros registros

print(wine.head())

# No de observaciones y tipo de datos

print(wine.info())

# Numero de Observaciones y Columnas

print(wine.shape)
matcorr = wine.iloc[:,~wine.columns.isin(['Id','quality'])].corr()

mask = np.zeros_like(matcorr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 9))

cmap = sns.color_palette("PRGn")

sns.heatmap(matcorr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True);

plt.show()
g = sns.PairGrid(wine.loc[:,["Total_Phenols","Flavanoids"]])

g.map_diag(plt.hist, histtype="step", linewidth=3)

g.map_offdiag(plt.scatter)
dist_sin = linkage(wine.loc[:,["Total_Phenols"]],method="single")

plt.figure(figsize=(18,14))

dendrogram(dist_sin, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("METODO SIMPLE DENDROGRAM",fontsize=12)

plt.show()

from scipy.cluster.hierarchy import fcluster

wine_1=wine.copy()



wine_1['2_clust']=fcluster(dist_sin,2, criterion='maxclust')

wine_1['3_clust']=fcluster(dist_sin,3, criterion='maxclust')

wine_1['4_clust']=fcluster(dist_sin,3, criterion='maxclust')



wine_1.head()
plt.figure(figsize=(24,4))



plt.suptitle("Wines",fontsize=18)



plt.subplot(1,4,1)

plt.title("K = 2",fontsize=12)

sns.scatterplot(x="Flavanoids",y="Total_Phenols", data=wine_1, hue="2_clust",palette="Paired")



plt.subplot(1,4,2)

plt.title("K = 3",fontsize=12)

sns.scatterplot(x="Flavanoids",y="Total_Phenols", data=wine_1, hue="3_clust",palette="Paired")



plt.subplot(1,4,3)

plt.title("K = 4",fontsize=12)

sns.scatterplot(x="Flavanoids",y="Total_Phenols", data=wine_1, hue="4_clust",palette="Paired")



plt.subplot(1,4,4)

plt.title("Vinos",fontsize=12)

sns.scatterplot(x="Flavanoids",y="Total_Phenols", data=wine_1, hue="Customer_Segment",palette="Paired")
plt.figure(figsize=(24,4))

plt.subplot(1,3,1)

plt.title("K = 2",fontsize=12)

sns.swarmplot(x="Customer_Segment",y="2_clust", data=wine_1, hue="Customer_Segment")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=12)

sns.swarmplot(x="Customer_Segment",y="3_clust", data=wine_1, hue="Customer_Segment")



plt.subplot(1,3,3)

plt.title("K = 4",fontsize=12)

sns.swarmplot(x="Customer_Segment",y="4_clust", data=wine_1, hue="Customer_Segment")



print(pd.crosstab(wine_1["Customer_Segment"],wine_1["4_clust"]))
dist_comp = linkage(wine.loc[:,["Total_Phenols"]],method="complete")



plt.figure(figsize=(18,6))

dendrogram(dist_comp, leaf_rotation=70)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("METODO DE DENDROGRAM ",fontsize=18) 

plt.show()
wine_2=wine.copy()

wine_2['2_clust']=fcluster(dist_comp,2, criterion='maxclust')

wine_2['3_clust']=fcluster(dist_comp,3, criterion='maxclust')

wine_2['4_clust']=fcluster(dist_comp,4, criterion='maxclust')

wine_2.head()
plt.figure(figsize=(18,5))



plt.suptitle("Metodo",fontsize=18)



plt.subplot(1,4,1)

plt.title("K = 2",fontsize=12)

sns.scatterplot(x="OD280",y="Total_Phenols", data=wine_2, hue="2_clust",palette="Paired")



plt.subplot(1,4,2)

plt.title("K = 3",fontsize=12)

sns.scatterplot(x="OD280",y="Total_Phenols", data=wine_2, hue="3_clust",palette="Paired")



plt.subplot(1,4,3)

plt.title("K = 4",fontsize=12)

sns.scatterplot(x="OD280",y="Total_Phenols", data=wine_2, hue="4_clust",palette="Paired")



plt.subplot(1,4,4)

plt.title("Customer_Segment",fontsize=12)

sns.scatterplot(x="OD280",y="Total_Phenols", data=wine_2, hue="Customer_Segment")
plt.figure(figsize=(18,4))

plt.subplot(1,3,1)

plt.title("K = 2",fontsize=12)

sns.swarmplot(x="Customer_Segment",y="2_clust", data=wine_2, hue="Customer_Segment")



plt.subplot(1,3,2)

plt.title("K = 3",fontsize=12)

sns.swarmplot(x="Customer_Segment",y="3_clust", data=wine_2, hue="Customer_Segment")



plt.subplot(1,3,3)

plt.title("K = 4",fontsize=12)

sns.swarmplot(x="Customer_Segment",y="4_clust", data=wine_2, hue="Customer_Segment")
print(pd.crosstab(wine_2["Customer_Segment"],wine_2["2_clust"]))

print(pd.crosstab(wine_2["Customer_Segment"],wine_2["3_clust"]))

print(pd.crosstab(wine_2["Customer_Segment"],wine_2["4_clust"]))