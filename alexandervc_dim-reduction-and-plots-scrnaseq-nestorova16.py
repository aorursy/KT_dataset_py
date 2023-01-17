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

import umap

df = pd.read_csv("/kaggle/input/single-cell-rna-seq-nestorova2016-mouse-hspc/nestorowa_corrected_log2_transformed_counts.txt", sep=' ',  )

df
X = df.values.copy()
y = pd.read_csv("/kaggle/input/single-cell-rna-seq-nestorova2016-mouse-hspc/nestorowa_corrected_population_annotation.txt", sep=' ')

y
y.iloc[:,0].unique()

# Create cell types markers



# First ones - loaded from anotations

df2 = df.join(y)

df2['celltype'].fillna('no_gate', inplace = True)

vec_cell_types_from_annotations = df2['celltype']

vec_cell_types_from_annotations



# Second ones - l

#  Extract some cell types markers from the cells ids: 

l = []# set()

for i in df.index:

    l.append(i[:4])

l = np.array(l)    

for m in np.unique(l):

    print(m,  (l==m).sum() )

vec_cell_types_from_dataframeindex = l    
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA



X = df.values.copy()



pca = PCA()

r = pca.fit_transform(X.copy())



plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_dataframeindex  )

plt.show()



plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )



fig = plt.figure(figsize = (15,7) )

fig.add_subplot(1, 2, 1) 

plt.plot(pca.singular_values_,'o-')

plt.title('Singular values')

fig.add_subplot(1, 2, 2) 

plt.plot(pca.explained_variance_ratio_,'o-')

plt.title('explained variance')
import matplotlib.pyplot as plt

import seaborn as sns

import umap



X = df.values



r = umap.UMAP().fit_transform(X.copy())

#plt.scatter(r[:,0],r[:,1],c = adata.obs['cell_types'].values )

plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_dataframeindex  )

plt.show()



plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )
import umap

import time





fig = plt.figure(figsize=(15,15))



perplexities = [5, 30, 50, 100]

c=0

for i,n_neighbors in enumerate( [5,200] ) :

  for min_dist in [0,0.5,1] :

    c += 1

    t0 = time.time()

    #tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)

    #r = tsne.fit_transform(X)

    r = umap.UMAP(n_neighbors= n_neighbors, min_dist = min_dist ).fit_transform(X)

    td = time.time()-t0

    print(td, 'secs passed')

    ax = fig.add_subplot(2, 3,c )

    sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )

    ax.set_title("n_neighbors=%d min_dist=%.1f.  %.2f secs" % (n_neighbors, min_dist, td ) )
from sklearn.manifold import TSNE



#r = TSNE(n_components=2).fit_transform(X)

#ax = fig.add_subplot(1, 2, 2  )

#ax.scatter(r[:,0],r[:,1],c = y)



plt.style.use('ggplot')



fig = plt.figure(figsize=(15,15))



t0 = time.time()

tsne = TSNE()

r = tsne.fit_transform(X)

td = time.time()-t0

print(td, 'secs passed')

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )



plt.show()
from sklearn.manifold import TSNE



#r = TSNE(n_components=2).fit_transform(X)

#ax = fig.add_subplot(1, 2, 2  )

#ax.scatter(r[:,0],r[:,1],c = y)



# plt.style.use('ggplot')



fig = plt.figure(figsize=(15,15))



perplexities = [5, 30, 100, 1000]

for i,perplexity in enumerate( perplexities ) :

  t0 = time.time()

  tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)

  r = tsne.fit_transform(X)

  #r = umap.UMAP().fit_transform(X)

  td = time.time()-t0

  print(td, 'secs passed')

  ax = fig.add_subplot(2, 2, i +1 )

  #ax.scatter(r[:,0],r[:,1],c = y)

  #sns.scatterplot(r[:,0], r[:,1])#, hue = y)

  sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )

  ax.set_title("Perplexity=%d .  %.2f secs passed" % (perplexity, td ) )
from sklearn.decomposition import FactorAnalysis



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

import time 



X = df.values



#pca = PCA()

transformer = FactorAnalysis(n_components=2, random_state=0)

t0 = time.time()

r = transformer.fit_transform(X.copy())

print(time.time() - t0 )



plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_dataframeindex  )

plt.show()



plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )
# Based on: 

# https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py

# See also:

# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#



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

methods['umap'] = umap.UMAP()

methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

methods['ICA'] = FastICA(n_components=n_components,         random_state=0)

methods['FA'] = FactorAnalysis(n_components=2, random_state=0)

methods['LLE'] = LLE(method='standard')

methods['Modified LLE'] = LLE(method='modified')

methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)

methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)

methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,

                                           n_neighbors=n_neighbors)

methods['NMF'] = NMF(n_components=n_components,  init='random', random_state=0) 

methods['RandProj'] = SparseRandomProjection(n_components=n_components, random_state=42)



rand_trees_embed = make_pipeline(RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5), TruncatedSVD(n_components=2) )

methods['RandTrees'] = rand_trees_embed

methods['LatDirAll'] = LatentDirichletAllocation(n_components=5,  random_state=0)

methods['LTSA'] = LLE(method='ltsa') # Fails on that dataset

methods['Hessian LLE'] = LLE(method='hessian') # Fails on that dataset

# transformer = NeighborhoodComponentsAnalysis(init='random',  n_components=2, random_state=0) # Cannot be applied since supervised - requires y 

# methods['LinDisA'] = LinearDiscriminantAnalysis(n_components=n_components)# Cannot be applied since supervised - requires y 





# Create figure

fig = plt.figure(figsize=(25, 16))



# Plot results

c = 0

for i, (label, method) in enumerate(methods.items()):

    if i == 0: # First plot - just to show legend 

        c += 1

        fig.add_subplot(4, 4 , c) 

        sns.scatterplot(x=X[:,0], y=X[:,1], hue = vec_cell_types_from_dataframeindex  )

        #sns.scatterplot(x=X[:,0], y=X[:,1], hue = vec_cell_types_from_annotations  )

        plt.title('First 2 coords' )

        

        

    t0 = time.time()

    try:

        r = method.fit_transform(X.copy())

    except:

        print('Got Exception', label )

        continue 

    t1 = time.time()

    print("%s: %.2g sec" % (label, t1 - t0))

    c+=1

    fig.add_subplot(4, 4 , c) 

    sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_dataframeindex  )

    #sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )

    plt.title(label )

    plt.legend('')



plt.show()

# Based on: 

# https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py

# See also:

# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#



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

methods['umap'] = umap.UMAP()

methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

methods['ICA'] = FastICA(n_components=n_components,         random_state=0)

methods['FA'] = FactorAnalysis(n_components=2, random_state=0)

methods['LLE'] = LLE(method='standard')

methods['Modified LLE'] = LLE(method='modified')

methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)

methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)

methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,

                                           n_neighbors=n_neighbors)

methods['NMF'] = NMF(n_components=n_components,  init='random', random_state=0) 

methods['RandProj'] = SparseRandomProjection(n_components=n_components, random_state=42)



rand_trees_embed = make_pipeline(RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5), TruncatedSVD(n_components=2) )

methods['RandTrees'] = rand_trees_embed

methods['LatDirAll'] = LatentDirichletAllocation(n_components=5,  random_state=0)

methods['LTSA'] = LLE(method='ltsa') # Fails on that dataset

methods['Hessian LLE'] = LLE(method='hessian') # Fails on that dataset

# transformer = NeighborhoodComponentsAnalysis(init='random',  n_components=2, random_state=0) # Cannot be applied since supervised - requires y 

# methods['LinDisA'] = LinearDiscriminantAnalysis(n_components=n_components)# Cannot be applied since supervised - requires y 





# Create figure

fig = plt.figure(figsize=(25, 16))



# Plot results

c = 0

for i, (label, method) in enumerate(methods.items()):

    if i == 0: # First plot - just to show legend 

        c += 1

        fig.add_subplot(4, 4 , c) 

        #sns.scatterplot(x=X[:,0], y=X[:,1], hue = vec_cell_types_from_dataframeindex  )

        sns.scatterplot(x=X[:,0], y=X[:,1], hue = vec_cell_types_from_annotations  )

        plt.title('First 2 coords' )

        

        

    t0 = time.time()

    try:

        r = method.fit_transform(X.copy())

    except:

        print('Got Exception', label )

        continue 

    t1 = time.time()

    print("%s: %.2g sec" % (label, t1 - t0))

    c+=1

    fig.add_subplot(4, 4 , c) 

    # sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_dataframeindex  )

    sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )

    plt.title(label )

    plt.legend('')



plt.show()
