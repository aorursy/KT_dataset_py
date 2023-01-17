import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import seaborn as sns

from math import ceil



%matplotlib inline
df_train = pd.read_csv('../input/kddbr-2019/public/training_dataset.csv', nrows=100_000)
df_train_labels = pd.read_csv('../input/kddbr-2019/public/training_data_labels.csv', index_col='scatterplotID')
df_train_labels['score'].hist(bins=21);
df_train['cluster'] = df_train['cluster'].astype('category')
d_scatters = dict(list(df_train.groupby('scatterplotID')))
n_plots = 48

n_cols = 4

n_rows = ceil(n_plots/n_cols)

cmap = plt.cm.Dark2



fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4), sharex=True, sharey=True)

axes = axes.ravel()



choosen_scatterplots_ids = np.random.choice(list(d_scatters.keys()), n_plots, replace=False)



# Sort by score

choosen_scatterplots_ids = df_train_labels.loc[choosen_scatterplots_ids, 'score'].sort_values().index



for i in range(n_plots):

    scatterplotID = choosen_scatterplots_ids[i]

    df_scatter = d_scatters[scatterplotID]

    

    x_pca = df_scatter[['signalX', 'signalY']].values# PCA(2).fit_transform(df_scatter[['signalX', 'signalY']])

    cluster = pd.Categorical(df_scatter['cluster'], df_train.cluster.cat.categories)

    score = df_train_labels.loc[scatterplotID, 'score']

#     sns.kdeplot(*x_pca.T, ax=axes[i], color=cmap(i), shade=True, shade_lowest=False)

    axes[i].scatter(*x_pca.T, cmap=cmap, c=cluster.codes)

    axes[i].set_title(f'Score: {score:.4f}')



fig.tight_layout()