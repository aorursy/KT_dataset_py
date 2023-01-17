# Here is standard kaggle code:



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
!pip install scanpy

import scanpy.api as sc

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from scipy.stats import mode

from collections import Counter



sc.settings.verbosity = 3

sc.set_figure_params(color_map='viridis')

sc.logging.print_versions()

results_file = './write/nestorowa.h5ad'

results_file_denoised = './write/nestorowa_denoised.h5ad'



# 

picture_size4scanpyplots = 180 # 180 - big size 

sc.settings.set_figure_params(dpi=picture_size4scanpyplots, frameon=False, figsize=(3, 3), facecolor='white')  # low dpi (dots per inch) yields small inline figures

# Load gene expression

fn = "/kaggle/input/single-cell-rna-seq-nestorova2016-mouse-hspc/nestorowa_corrected_log2_transformed_counts.txt"

#adata = sc.read('./data/nestorowa_corrected_log2_transformed_counts.txt', cache=True)

adata = sc.read(fn, cache=True)





#Load cell type annotation

fn = "/kaggle/input/single-cell-rna-seq-nestorova2016-mouse-hspc/nestorowa_corrected_population_annotation.txt"

#cell_types = pd.read_csv('./data/nestorowa_corrected_population_annotation.txt', delimiter=' ')

cell_types = pd.read_csv(fn, delimiter=' ')



# replace with shorter names

acronyms = {'ESLAM': 'Stem', 'Erythroid': 'Ery', 'Megakaryocytes': 'Mk', 'Basophils': 'Baso',

            'Neutrophils': 'Neu', 'Monocytes': 'Mo', 'Bcell': 'B'}

# add this cell type information

cell_types = [acronyms[cell_types.loc[cell_id, 'celltype']]

              if cell_id in cell_types.index else 'no_gate' for cell_id in adata.obs_names]

adata.obs['cell_types'] = cell_types
# Main data are stored in numpy array: adata.X - 1645 cells x 3991 gene

print(adata.X.shape, type(adata.X))

pd.DataFrame(adata.X).describe()
# adata.obs - cell type annotation , it is mostly "no_gate" and 8-10 types for 'ESLAM': 'Stem', 'Erythroid': 'Ery', 'Megakaryocytes': 'Mk', 'Basophils': 'Baso',

#            'Neutrophils': 'Neu', 'Monocytes': 'Mo', 'Bcell': 'B'

print(type(adata.obs), adata.obs.shape)

#adata.obs.head()



adata.obs['cell_types'].value_counts()


print(type(adata.var))

for k in range(0,4000,100):

    print('Genes',k,'-',k+100)

    print(adata.var.index[k:k+100])
import matplotlib.pyplot as plt

import seaborn as sns 
from sklearn.decomposition import PCA



r = PCA().fit_transform(adata.X.copy())

#plt.scatter(r[:,0],r[:,1],c = adata.obs['cell_types'].values )

plt.figure(figsize = (15,10) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = adata.obs['cell_types'].values )

import umap



r = umap.UMAP().fit_transform(adata.X.copy())

#plt.scatter(r[:,0],r[:,1],c = adata.obs['cell_types'].values )

plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = adata.obs['cell_types'].values )

sc.settings.set_figure_params(dpi=180, frameon=False, figsize=(3, 3), facecolor='white')  # low dpi (dots per inch) yields small inline figures



sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)

sc.tl.draw_graph(adata, layout='fa', random_state=1)

sc.pl.draw_graph(adata, color='cell_types')



import igraph

!pip install louvain
sc.tl.louvain(adata, resolution=1)

sc.tl.draw_graph(adata, layout='fa', random_state=1)

sc.pl.draw_graph(adata, color=['louvain', 'cell_types', 'Hba-a2', 'Elane', 'Irf8'], legend_loc='on data')
