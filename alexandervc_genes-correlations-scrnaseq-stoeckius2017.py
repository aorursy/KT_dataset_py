# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import time



import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('/kaggle/input/single-cell-rna-seq-from-stoeckius-et-al-2017/GSE100866_CD8_merged-RNA_umi.csv', index_col = 0)

# The dataset is single cell rna seq from the Cite-seq paper. 

# Cells were classified by flow cytometry as low, medium, or high cd8 expression cells (columns), with rows corresponding to RNA counts. 1,774 cells in total by 11757 genes. 

# Stoeckius et al 2017.



df = df.T

df
X = df.values

X.shape
t0 = time.time()

corr_matr = np.corrcoef(X.T) # Hint - use numpy , pandas is MUCH SLOWER   (df.corr() )

print(time.time() - t0, 'seconds passed')

print(np.min(corr_matr ), 'minimal correlation' )

corr_matr_abs = np.abs( corr_matr )

print(np.mean(corr_matr_abs ), 'average absolute correlation' )

print(np.median(corr_matr_abs), 'median absolute correlation' )

print(np.min(corr_matr_abs ), 'min absolute correlation' )

print(np.std(corr_matr_abs ), 'std absolute correlation' )
v = corr_matr.flatten()

plt.figure(figsize=(14,8))

t0 = time.time()

plt.hist(v, bins = 50)

plt.title('correlation coefficients distribution')

plt.show()

print(time.time() - t0, 'seconds passed')



print(np.min(corr_matr ), 'minimal correlation' )

print(np.mean(corr_matr_abs ), 'average absolute correlation' )

print(np.median(corr_matr_abs), 'median absolute correlation' )

print(np.min(corr_matr_abs ), 'min absolute correlation' )

print(np.std(corr_matr_abs ), 'std absolute correlation' )

for t in [0.5,0.6, 0.7,0.8,0.9,0.95,0.97,0.98,.99]:

    print( ((np.abs(v) < 0.99999999) & (np.abs(v) > t)).sum()/2 , 'number of pairs correlated more than', t  )

v.shape




plt.figure(figsize=(14,8))

t0 = time.time()

sns.heatmap(corr_matr_abs).set_title('Correlation (abs) heatmap')

print(time.time() - t0, 'seconds passed')







import igraph



corr_matr_abs_bool = corr_matr_abs > 0.6

corr_matr_abs_bool = corr_matr_abs_bool# [:772 ,:772 ]

corr_matr_abs_bool = np.triu(corr_matr_abs_bool,1) # Take upper triangular part 

g = igraph.Graph().Adjacency(corr_matr_abs_bool.tolist())

g.to_undirected(mode = 'collapse')

print( corr_matr_abs_bool.astype(int) )



print('Number of nodes ', g.vcount())

print('Number of edges ', g.ecount() )

print('Number of weakly connected compoenents', len( g.clusters(mode='WEAK')))





print('Sizes of connected components large than 5 nodes')

c = 0

for t in list(g.clusters(mode='WEAK') ):

    if len(t) <= 5: continue

    c+=1 

    print(len(t) )

print('count components large than 5 nodes:', c )

    

visual_style = {}

visual_style["vertex_color"] = ['green' for v in g.vs]

#visual_style["vertex_label"] = range(g.vcount()) 

visual_style["vertex_size"] = 2

igraph.plot(g,bbox = (800,500), **visual_style )




# Plot the largest component separately 

for t in list(g.clusters(mode='WEAK') ):

    if len(t) <= 30: continue

    print(t)

    g2 = g.subgraph(t)

    index_save = t.copy()

    

print('Number of nodes ', g2.vcount())

print('Number of edges ', g2.ecount() )    

print(df.columns[index_save])    

igraph.plot(g2,bbox = (800,200), **visual_style )


