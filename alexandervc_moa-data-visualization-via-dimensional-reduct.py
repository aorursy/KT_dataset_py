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
y = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv',index_col = 0 )

y
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



X_original_save = X.copy()

print(X.shape)



y_sum = y.sum(axis = 1)

df['y_sum']=y_sum

print(y_sum.shape)

y_sum.value_counts()



from sklearn.decomposition import PCA

import time 

import matplotlib.pyplot as plt

import seaborn as sns



pca = PCA()

t0 = time.time()

r = pca.fit_transform(X.copy())

print(time.time()-t0, 'secs passed for PCA')





fig = plt.figure(figsize = (15,7) )

c = 0

for f in ['cp_dose', 'cp_type','cp_time', 'y_sum']:

    c+=1; fig.add_subplot(1, 4 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = df[f]  )

    plt.title('Colored by '+f)

plt.show()



fig = plt.figure(figsize = (15,7) )

fig.add_subplot(1, 2, 1) 

plt.plot(pca.singular_values_,'o-')

plt.title('Singular values')

fig.add_subplot(1, 2, 2) 

plt.plot(pca.explained_variance_ratio_,'o-')

plt.title('explained variance')
import umap



t0 = time.time()

r = umap.UMAP().fit_transform(X.copy())

print(time.time()-t0, 'secs passed for umap')





fig = plt.figure(figsize = (15,7) )

c = 0

for f in ['cp_dose', 'cp_type','cp_time','y_sum']:

    c+=1; fig.add_subplot(1, 4 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = df[f]  )

plt.show()
# Based on: 

# https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py

# See also:

# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#







# To speed-up reduce dimensions by PCA first

X_save = X.copy( )

#r = pca.fit_transform(X)

#X = r[:1000,:20]







import umap 

from sklearn import manifold

from sklearn.decomposition import PCA

from sklearn.decomposition import FactorAnalysis

from sklearn.decomposition import NMF

from sklearn.decomposition import FastICA

from sklearn.decomposition import FactorAnalysis

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.ensemble import RandomTreesEmbedding

from sklearn.random_projection import SparseRandomProjection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.pipeline import make_pipeline

from sklearn.decomposition import TruncatedSVD





from collections import OrderedDict

from functools import partial

from matplotlib.ticker import NullFormatter





n_neighbors = 10

n_components = 2

# Set-up manifold methods

LLE = partial(manifold.LocallyLinearEmbedding,

              n_neighbors, n_components, eigen_solver='auto')



methods = OrderedDict()

methods['PCA'] = PCA()

methods['umap'] = umap.UMAP(n_components = n_components)

methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

methods['ICA'] = FastICA(n_components=n_components,         random_state=0)

methods['FA'] = FactorAnalysis(n_components=n_components, random_state=0)

methods['LLE'] = LLE(method='standard')

methods['Modified LLE'] = LLE(method='modified')

methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)

methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)

methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,

                                           n_neighbors=n_neighbors)

methods['NMF'] = NMF(n_components=n_components,  init='random', random_state=0) 

methods['RandProj'] = SparseRandomProjection(n_components=n_components, random_state=42)



rand_trees_embed = make_pipeline(RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5), TruncatedSVD(n_components=n_components) )

methods['RandTrees'] = rand_trees_embed

methods['LatDirAll'] = LatentDirichletAllocation(n_components=n_components,  random_state=0)

methods['LTSA'] = LLE(method='ltsa') 

methods['Hessian LLE'] = LLE(method='hessian') 



list_fast_methods = ['PCA','umap','FA', 'ICA','NMF','RandProj','RandTrees']

list_slow_methods = ['t-SNE','LLE','Modified LLE','Isomap','MDS','SE','LatDirAll','LTSA','Hessian LLE']



# transformer = NeighborhoodComponentsAnalysis(init='random',  n_components=2, random_state=0) # Cannot be applied since supervised - requires y 

# methods['LinDisA'] = LinearDiscriminantAnalysis(n_components=n_components)# Cannot be applied since supervised - requires y 





# Create figure

fig = plt.figure(figsize=(25, 16))



# Plot results

c = 0

for i, (label, method) in enumerate(methods.items()):

    if label not in  list_fast_methods :

        continue

        

    t0 = time.time()

    try:

        r = method.fit_transform(X.copy())

    except:

        print('Got Exception', label )

        continue 

    t1 = time.time()

    print("%s: %.2g sec" % (label, t1 - t0))

    c+=1

    fig.add_subplot(2, 3 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1],hue =  df['cp_time'])

    plt.title(label )

    plt.legend('')



plt.show()

X = X_save.copy()

# To speed-up reduce dimensions by PCA first and cut the size

X_save = X.copy( )

r = pca.fit_transform(X)

i_cut = 5000

X = r[:i_cut,:50]





list_slow_methods = ['t-SNE','LLE','Modified LLE','Isomap','MDS','SE','LatDirAll','LTSA','Hessian LLE']



# transformer = NeighborhoodComponentsAnalysis(init='random',  n_components=2, random_state=0) # Cannot be applied since supervised - requires y 

# methods['LinDisA'] = LinearDiscriminantAnalysis(n_components=n_components)# Cannot be applied since supervised - requires y 





# Create figure

fig = plt.figure(figsize=(25, 4))



# Plot results

c = 0

for i, (label, method) in enumerate(methods.items()):

    #if label not in list_slow_methods: # list_fast_methods :

    #    continue

        

    t0 = time.time()

    try:

        r = method.fit_transform(X.copy())

    except:

        print('Got Exception', label )

        continue 

    t1 = time.time()

    print("%s: %.2g sec" % (label, t1 - t0))

    c+=1

    fig.add_subplot(1, 4 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1],hue =  df['cp_time'][:i_cut])

    plt.title(label )

    plt.legend('')

    if c%4 == 0:

        c = 0

        plt.show()

        fig = plt.figure(figsize=(25, 4))

        

    



plt.show()

X = X_save.copy()

X_save.shape
from sklearn.decomposition import PCA

import time 

import matplotlib.pyplot as plt

import seaborn as sns



# Preliminary reduce X to speed up:

r = pca.fit_transform(X.copy())

i_cut = 5000

r = r[:i_cut,:50]





n_neighbors = 10

n_components = 2

method = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto', method='standard')

t0 = time.time()

r = method.fit_transform(r)

t1 = time.time()

print("%s: %.2g sec" % ('LLE', t1 - t0))



fig = plt.figure(figsize = (15,7) )

c = 0

for f in ['cp_dose', 'cp_type','cp_time', 'y_sum']:

    c+=1; fig.add_subplot(1, 4 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = df[f][:i_cut]  )

    plt.title('Colored by '+f)

plt.show()

# Apply transform to full dataset 



r = pca.transform(X.copy() )

r = r[:,:50]



t0 = time.time()

r = method.transform(r)

print(time.time()-t0,'seconds passed')



fig = plt.figure(figsize = (15,7) )

c = 0

for f in ['cp_dose', 'cp_type','cp_time', 'y_sum']:

    c+=1; fig.add_subplot(1, 4 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = df[f]  )

    plt.title('Colored by '+f)

plt.show()

df[f].value_counts() # strange that seaborn plots only 0,2,5,7 values