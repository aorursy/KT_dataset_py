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



df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv',index_col = 0)  

df
df_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv',index_col = 0)  

df_test
y = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv',index_col = 0 )

y
y_additional = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv',index_col = 0 )

y_additional
y.sum(axis = 1).value_counts()
y_additional.sum(axis = 1).value_counts()
mode_which_part_to_process = 'full'

if mode_which_part_to_process == 'full':

    # consider only gene expression part 

    X = df[[c for c in df.columns if ('c-' in c) or ('g-' in c)]].values

if mode_which_part_to_process == 'genes':

    # consider only gene expression part 

    X = df[[c for c in df.columns if 'g-' in c]].values

if mode_which_part_to_process == 'c':

    # consider only gene expression part 

    X = df[[c for c in df.columns if 'c-' in c]].values



print(len([c for c in df.columns if 'g-' in c] ), 'genes count ')

X_original_save = X.copy()

print(X.shape)
t0 = time.time()

corr_matr = np.corrcoef(X.T) # Hint - use numpy , pandas is MUCH SLOWER   (df.corr() )

print(time.time() - t0, 'seconds passed')

print(np.min(corr_matr ), 'minimal correlation' )

corr_matr_abs = np.abs( corr_matr )

print(np.mean(corr_matr_abs ), 'average absolute correlation' )

print(np.median(corr_matr_abs), 'median absolute correlation' )

print(np.min(corr_matr_abs ), 'min absolute correlation' )

print(np.std(corr_matr_abs ), 'std absolute correlation' )





corr_matr.shape


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





v.shape
plt.figure(figsize=(14,8))

t0 = time.time()

sns.heatmap(corr_matr_abs).set_title('Correlation (abs) heatmap')

print(time.time() - t0, 'seconds passed')

import igraph
corr_matr_abs_bool = corr_matr_abs > 0.78

corr_matr_abs_bool = corr_matr_abs_bool[:772 ,:772 ]

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

print('count components large than one node:', c )

    

visual_style = {}

visual_style["vertex_color"] = ['green' for v in g.vs]

#visual_style["vertex_label"] = range(g.vcount()) 

visual_style["vertex_size"] = 2

igraph.plot(g,bbox = (800,500), **visual_style )

# Plot the largest component separately 

for t in list(g.clusters(mode='WEAK') ):

    if len(t) <= 5: continue

    print(t)

    g2 = g.subgraph(t)

igraph.plot(g2,bbox = (800,200), **visual_style )

    
# 0.7 -> 116

# 0.6 -> 243

# 0.5 -> 381

# 0.4  -> 548

# 0.3 -> 730

corr_matr_abs_bool = corr_matr_abs > 0.6



corr_matr_abs_bool = corr_matr_abs_bool[:772 ,:772 ]

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

print('count components large than one node:', c )

    

visual_style = {}

visual_style["vertex_color"] = ['red' for v in g.vs]

#visual_style["vertex_label"] = range(g.vcount()) 

visual_style["vertex_size"] = 15

igraph.plot(g,bbox = (800,500), **visual_style )

# Plot the largest component separately 

for t in list(g.clusters(mode='WEAK') ):

    if len(t) <= 5: continue

    print(t)

    g2 = g.subgraph(t)

    

print('Number of nodes ', g2.vcount())

print('Number of edges ', g2.ecount() )



layout = g2.layout('tree') # 'drl') # 'grid_fr') #'fr') # 'circle') # "kamada_kawai")    

visual_style["vertex_size"] = 3

igraph.plot(g2,bbox = (800,800), **visual_style, layout = layout)
verbose = 0

df_stat = pd.DataFrame() # dict_save_largest_component_size = {} 

i = 0

for correlation_threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4] :

    t0 = time.time()

    print()

    print(correlation_threshold , 'correlation_threshold ')

    corr_matr_abs_bool = corr_matr_abs > correlation_threshold

    corr_matr_abs_bool = corr_matr_abs_bool[:772 ,:772 ] # Restrict to  genes part 

    corr_matr_abs_bool = np.triu(corr_matr_abs_bool,1) # Take upper triangular part 

    g = igraph.Graph().Adjacency(corr_matr_abs_bool.tolist())

    g.to_undirected(mode = 'collapse')

    if verbose >= 10:

        print( corr_matr_abs_bool.astype(int) )

        print('Number of nodes ', g.vcount())

        print('Number of edges ', g.ecount() )

        print('Number of weakly connected compoenents', len( g.clusters(mode='WEAK')))





    list_clusters_nodes_lists = list( g.clusters(mode='WEAK') )

    list_clusers_size = [len(t) for t in list_clusters_nodes_lists ]

    list_clusers_size = np.sort(list_clusers_size)[::-1]

    print('Top 5 cluster sizes:', list_clusers_size[:5] , 'seconds passed:', np.round(time.time()-t0 , 2))

    #dict_save_largest_component_size[correlation_threshold ] = list_clusers_size[0]

    for t  in list_clusters_nodes_lists:

        if len(t) == list_clusers_size[0]:

            print('50 Genes in largest correlated group:')

            print(df.columns[t[:50]])

    i += 1

    df_stat.loc[i,'correlation threshold'] = correlation_threshold

    df_stat.loc[i,'Largest Component Size'] = list_clusers_size[0]

    df_stat.loc[i,'Second Component Size'] = list_clusers_size[1]

    

df_stat

df_stat