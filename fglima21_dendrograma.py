# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dados = pd.read_csv("../input/data.csv")

dados.head()
dados.columns
base_dendro = dados[['Strength','Finishing']].copy()

base_dendro.dropna(inplace=True)

base_dendro = base_dendro.head(100)

base_dendro.head(10)
base_dendro.plot.scatter(x='Strength', y='Finishing')
from scipy.cluster.hierarchy import linkage, fcluster # linkage criar matriz de distãncia

                                                      # fcluster roluta cluster ao dado



distance_matrix = linkage(base_dendro, 

                                method='single', # como calcular a proximidade dos clusteres

                                metric='euclidean', # métrica de distância

                                optimal_ordering=False) # ordenação dos data points



base_dendro['cluster'] = fcluster(distance_matrix, # output do método linkage()

                                 3, # numero de clusteres

                                 criterion='maxclust') # o critério do limiar dos clusteres

from scipy.cluster.hierarchy import dendrogram

dn = dendrogram(distance_matrix) # é o resultado da função linkage

plt.show()
from matplotlib import pyplot as plt

import seaborn as sns



sns.scatterplot(x='Strength',

                y='Finishing',

                hue='cluster',

                data=base_dendro,

                legend=False)



plt.show()
base_dendro.cluster.value_counts()
base_dendro_b = dados[['Agility','Stamina']].copy()

base_dendro_b = base_dendro_b.dropna()

base_dendro_b = base_dendro_b.head(100)
base_dendro_b.plot.scatter(x='Agility',y='Stamina')
dendrogram(linkage(base_dendro_b, method='single') )

plt.show()
Z = linkage(base_dendro_b, method='single')

base_dendro_b['single_cluster_inco'] = fcluster(Z,3,criterion='inconsistent')

base_dendro_b['single_cluster_dist'] = fcluster(Z,3,criterion='distance')

base_dendro_b['single_cluster_maxc'] = fcluster(Z,3,criterion='maxclust')

# base_dendro_b['single_cluster_mono'] = fcluster(Z,3,criterion='monocrit')

# base_dendro_b['single_cluster_max_mono'] = fcluster(Z,3,criterion='maxclust_monocrit')
sns.scatterplot(x='Agility',

                y='Stamina',

                hue='single_cluster_inco',

                data=base_dendro_b,

                legend=False)



plt.show()
sns.scatterplot(x='Agility',

                y='Stamina',

                hue='single_cluster_dist',

                data=base_dendro_b,

                legend=False)



plt.show()
tipo = 'single_cluster_maxc'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue='single_cluster_maxc',

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
dendrogram(linkage(base_dendro_b, method='complete') )

plt.show()
Z = linkage(base_dendro_b, method='complete')

base_dendro_b['complete_cluster_inco'] = fcluster(Z,3,criterion='inconsistent')

base_dendro_b['complete_cluster_dist'] = fcluster(Z,3,criterion='distance')

base_dendro_b['complete_cluster_maxc'] = fcluster(Z,3,criterion='maxclust')

#base_dendro_b['complete_cluster_mono'] = fcluster(Z,3,criterion='monocrit')

#base_dendro_b['complete_cluster_max_mono'] = fcluster(Z,3,criterion='maxclust_monocrit')
tipo = 'complete_cluster_inco'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue = tipo,

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
tipo = 'complete_cluster_dist'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue='complete_cluster_dist',

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
tipo = 'complete_cluster_maxc'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue='complete_cluster_maxc',

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
dendrogram(linkage(base_dendro_b, method='average') )

plt.show()
Z = linkage(base_dendro_b, method='average')

base_dendro_b['average_cluster_inco'] = fcluster(Z,3,criterion='inconsistent')

base_dendro_b['average_cluster_dist'] = fcluster(Z,3,criterion='distance')

base_dendro_b['average_cluster_maxc'] = fcluster(Z,3,criterion='maxclust')

#base_dendro_b['average_cluster_mono'] = fcluster(Z,3,criterion='monocrit')

#base_dendro_b['average_cluster_max_mono'] = fcluster(Z,3,criterion='maxclust_monocrit')
tipo = 'average_cluster_inco'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue=tipo,

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
tipo = 'average_cluster_dist'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue=tipo,

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
tipo = 'average_cluster_maxc'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue=tipo,

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
dendrogram(linkage(base_dendro_b, method='weighted') )

plt.show()
Z = linkage(base_dendro_b, method='weighted')

base_dendro_b['weighted_cluster_inco'] = fcluster(Z,3,criterion='inconsistent')

base_dendro_b['weighted_cluster_dist'] = fcluster(Z,3,criterion='distance')

base_dendro_b['weighted_cluster_maxc'] = fcluster(Z,3,criterion='maxclust')

#base_dendro_b['weighted_cluster_mono'] = fcluster(Z,3,criterion='monocrit')

#base_dendro_b['weighted_cluster_max_mono'] = fcluster(Z,3,criterion='maxclust_monocrit')
tipo = 'average_cluster_maxc'

sns.scatterplot(x='Agility',

                y='Stamina',

                hue=tipo,

                data=base_dendro_b,

                legend=False)



plt.show()

base_dendro_b[tipo].value_counts()
dendrogram(linkage(base_dendro_b, method='centroid') )

plt.show()
dendrogram(linkage(base_dendro_b, method='median') )

plt.show()
dendrogram(linkage(base_dendro_b, method='ward') )

plt.show()