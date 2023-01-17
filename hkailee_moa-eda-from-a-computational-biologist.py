# # Install libraries/pacakges
!conda install seaborn scikit-learn statsmodels numba pytables -y
!conda install -c conda-forge python-igraph leidenalg -y
!pip install quanp
!pip install MulticoreTSNE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pl
import quanp as qp

from IPython.display import display
from matplotlib import rcParams

# setting visualization/logging parameters
pd.set_option('display.max_columns', None)
qp.set_figure_params(dpi=100, color_map = 'viridis_r')
qp.settings.verbosity = 1
qp.logging.print_versions()
# Loading pandas dataframe as anndata 
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv', index_col=0)

# Get lists of genes and cell viabilities, respectively
train_genes = [s for s in train_features.columns if "g-" in s]
train_cellvia = [s for s in train_features.columns if "c-" in s]
# Loading pandas dataframe as anndata 
adata_genes = qp.AnnData(train_features[train_genes])
adata_cellvia = qp.AnnData(train_features[train_cellvia])

# add a new `.obs` column for additional categorical features
adata_genes.obs['cp_type'] = train_features['cp_type']
adata_genes.obs['cp_time'] = train_features['cp_time']
adata_genes.obs['cp_dose'] = train_features['cp_dose']
adata_cellvia.obs['cp_type'] = train_features['cp_type']
adata_cellvia.obs['cp_time'] = train_features['cp_time']
adata_cellvia.obs['cp_dose'] = train_features['cp_dose']
train_features[train_genes].describe()
rcParams['figure.figsize'] = 12, 8
qp.tl.pca(adata_genes, svd_solver='auto');
qp.pl.pca(adata_genes, 
          color=['cp_type', 'cp_time', 'cp_dose'], 
          size=50, 
          ncols=2);
qp.pl.pca_variance_ratio(adata_genes, n_pcs=20)
qp.pp.neighbors(adata_genes, n_neighbors=30, n_pcs=10); # 30 nearest neighbors and only consider the first 10 pcs
qp.tl.leiden(adata_genes);
rcParams['figure.figsize'] = 12, 8
qp.tl.tsne(adata_genes, n_pcs=10); # only consider the first 10 pcs

qp.pl.tsne(adata_genes, color=['leiden', 'cp_type', 'cp_time', 'cp_dose'], 
           legend_loc='on data', ncols=2)
rcParams['figure.figsize'] = 8,6
qp.tl.paga(adata_genes)
qp.pl.paga(adata_genes, plot=True)
rcParams['figure.figsize'] = 10, 6
qp.tl.umap(adata_genes, init_pos='paga')
qp.pl.umap(adata_genes, color=['leiden', 'cp_type', 'cp_time', 'cp_dose'], 
           legend_loc='on data', ncols=2)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning);

rcParams['figure.figsize'] = 6,8;
qp.tl.rank_features_groups(adata_genes, 'leiden', method='wilcoxon');
qp.pl.rank_features_groups(adata_genes, n_features=30, sharey=False)
qp.tl.dendrogram(adata_genes, 'leiden', var_names=adata_genes.var_names);
qp.pl.rank_features_groups_matrixplot(adata_genes, n_features=5, use_raw=False, 
                                      cmap='bwr'); # choose only top 5 features
rcParams['figure.figsize'] = 8,8;
qp.pl.umap(adata_genes, color=['leiden', 'g-744', 'g-243', 'g-712', 'g-417', 'g-731', 'g-166', 'g-167', 'g-708', 'g-168', 'g-456'], 
           legend_loc='on data', ncols=3, cmap='bwr')
rcParams['figure.figsize'] = 8,8
qp.pl.rank_features_groups_heatmap(adata_genes, n_features=5, use_raw=False, 
                                   vmin=-5, vmax=5, cmap='bwr')
train_features[train_cellvia].describe()
rcParams['figure.figsize'] = 8, 5
qp.tl.pca(adata_cellvia, svd_solver='auto');
qp.pl.pca(adata_cellvia, color=['cp_type', 'cp_time', 'cp_dose'], size=50);
qp.pl.pca_variance_ratio(adata_cellvia, n_pcs=20)
qp.pp.neighbors(adata_cellvia, n_neighbors=100, n_pcs=5); # only consider the first 5 pcs
qp.tl.leiden(adata_cellvia, resolution=0.5);
rcParams['figure.figsize'] = 12, 8
qp.tl.tsne(adata_cellvia, n_pcs=5); # only consider the first 5 pcs
qp.pl.tsne(adata_cellvia, color=['leiden', 'cp_type', 'cp_time', 'cp_dose'], 
           legend_loc='on data', ncols=2)
rcParams['figure.figsize'] = 8,6
qp.tl.paga(adata_cellvia)
qp.pl.paga(adata_cellvia, plot=True)
rcParams['figure.figsize'] = 10, 6
qp.tl.umap(adata_cellvia, init_pos='paga')
qp.pl.umap(adata_cellvia, color=['leiden', 'cp_type', 'cp_time', 'cp_dose'], 
           legend_loc='on data', ncols=2)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

rcParams['figure.figsize'] = 6,8;
qp.tl.rank_features_groups(adata_cellvia, 'leiden', method='wilcoxon');
qp.pl.rank_features_groups(adata_cellvia, n_features=30, sharey=False)
qp.tl.dendrogram(adata_cellvia, 'leiden', var_names=adata_cellvia.var_names);
qp.pl.matrixplot(adata_cellvia, adata_cellvia.var_names, 'leiden', dendrogram=True, cmap='RdBu_r')
rcParams['figure.figsize'] = 8,8;
qp.pl.umap(adata_cellvia, color=['leiden', 'c-0', 'c-1', 'c-2', 'c-3', 'c-22'], 
           legend_loc='on data', ncols=3, cmap='bwr')