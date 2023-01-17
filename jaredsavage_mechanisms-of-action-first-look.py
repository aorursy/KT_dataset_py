import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

%matplotlib inline
df_test_features = pd.read_csv('../input/lish-moa/test_features.csv')

df_train_features = pd.read_csv('../input/lish-moa/train_features.csv')

df_train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

df_train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

print(f'Train features:\t\t\t{df_train_features.shape}\nTest features:\t\t\t{df_test_features.shape}\nTrain targets (scored):\t\t{df_train_targets_scored.shape}\nTrain targets (non-scored):\t{df_train_targets_nonscored.shape}')

df_train_features.head(10)
df_train_targets_scored.head(10)
df_train_targets_nonscored.head(10)
cols_match_1 = df_train_targets_scored['sig_id']==df_train_features['sig_id']

cols_match_2 = df_train_targets_nonscored['sig_id']==df_train_features['sig_id']

cols_match_1.value_counts(),cols_match_2.value_counts()
df_train_features[['g-0','g-1','g-2','g-3','g-4','g-5','g-6','g-7','g-8','g-9']].describe()
df_train_features[['c-0','c-1','c-2','c-3','c-4','c-5','c-6','c-7','c-8','c-9']].describe()
cell_cols = df_train_features.iloc[:,-100:]

#cell_cols['sig_id'] = df_train_features['sig_id']



def draw_histograms(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(18,30), dpi= 100, facecolor='w', edgecolor='k')

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=20,ax=ax)

        ax.set_title(var_name)

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()



draw_histograms(cell_cols, cell_cols.columns, 20, 5)
gene_cols = df_train_features.iloc[:,4:104]

#cell_cols['sig_id'] = df_train_features['sig_id']



def draw_histograms(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(18,30), dpi= 100, facecolor='w', edgecolor='k')

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=20,ax=ax)

        ax.set_title(var_name)

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()



draw_histograms(gene_cols, gene_cols.columns, 20, 5)