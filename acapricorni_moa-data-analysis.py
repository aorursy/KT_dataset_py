import os

import pickle

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE
data_path = '../input/lish-moa'

! ls {data_path}
features_file = 'train_features.csv'

targets_file = 'train_targets_scored.csv'

no_targets_file = 'train_targets_nonscored.csv'
df_features = pd.read_csv(os.path.join(data_path, features_file))

df_targets = pd.read_csv(os.path.join(data_path, targets_file))

df_no_targets = pd.read_csv(os.path.join(data_path, no_targets_file))
# merge dataframes

df_data = df_features.merge(df_targets, how='left', on='sig_id', validate='one_to_one')

df_data = df_data.merge(df_no_targets, how='left', on='sig_id', validate='one_to_one')

df_data.head(5)
# keep columns names lists

# columns names = 'sig_id' + 'cp_type' + features_quali + features_quanti + scored_targets + no_scored_targets

scored_targets =  list(set(df_targets.columns) - set(['sig_id']))

no_scored_targets = list(set(df_no_targets.columns) - set(['sig_id']))

features_quali = ['cp_time', 'cp_dose']

features_quanti = list(set(df_data.columns) 

                       - set(scored_targets) 

                       - set(no_scored_targets)

                       - set(features_quali)

                       - set(['sig_id', 'cp_type']))

print('Scored targets count : {}'.format(len(scored_targets)))

print('No scored targets count : {}'.format(len(no_scored_targets)))

print('Features quali count : {}'.format(len(features_quali)))

print('Features quanti count : {}'.format(len(features_quanti)))
# separate features_quanti : gene expression and cell viability features

cells = [feature_name for feature_name in features_quanti if feature_name.find('c-') != -1]

genes = [feature_name for feature_name in features_quanti if feature_name.find('g-') != -1]

print('Features genes count : {}'.format(len(genes)))

print('Features cells count : {}'.format(len(cells)))
# shape

df_data.shape
# check sig_id is unique

test = df_data['sig_id'].is_unique

print('sig_id unique : {}'.format(test))
# check there are no MoA in 'control' test

moa_count = df_data[df_data['cp_type'] == 'ctl_vehicle'][scored_targets].sum().sum()

print('MoA count (control test) : {}'.format(moa_count))
# check nan

test = df_data.isnull().values.any()

print('Missing data : {}'.format(test))
# separate control and compound

df_compound = df_data[df_data['cp_type'] == 'trt_cp']

df_control = df_data[df_data['cp_type'] == 'ctl_vehicle']

print('Compound shape : {}'.format(df_compound.shape))

print('Control shape : {}'.format(df_control.shape))
df_control_features = df_control[features_quali + features_quanti]

df_control_features.head(3)
# plot dose and time exposure distribution

sns.countplot(x='cp_time', hue='cp_dose', data=df_control_features)
df_control_cell = df_control[features_quali + cells]

df_control_cell.head(3)
# get variance for each cell and identify cells with min and max variance

cell_std = df_control_cell[cells].std().sort_values(ascending=False)

cell_std_max_min = [cell_std.iloc[[0]].index,

                    cell_std.iloc[[-1]].index]

cell_std_max_min
# plot distribution of sample for cell with MIN variance viability

g = sns.FacetGrid(df_control_cell, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'c-98')

g.add_legend()
# plot distribution of sample for cell with MAX variance viability

g = sns.FacetGrid(df_control_cell, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'c-18')

g.add_legend()
df_control_gene = df_control[features_quali + genes]

df_control_gene.head(3)
# get variance for each gene and identify genes with min and max variance

gene_std = df_control_gene[genes].std().sort_values(ascending=False)

gene_std_max_min = [gene_std.iloc[[0]].index,

                    gene_std.iloc[[-1]].index]

gene_std_max_min
# plot distribution of sample for gene with MIN variance viability

g = sns.FacetGrid(df_control_gene, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'g-307', bw=0.1)

g.add_legend()
# plot distribution of gene with MAX variance viability

g = sns.FacetGrid(df_control_gene, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'g-370', bw=0.1)

g.add_legend()
df_compound_features = df_compound[features_quali + features_quanti]

df_compound_features.head(3)
# plot dose and time exposure distribution

sns.countplot(x='cp_time', hue='cp_dose', data=df_compound_features)
df_compound_cell = df_compound[features_quali + cells]

df_compound_cell.head(3)
# get variance for each cell and identify cells with min and max variance

cell_std = df_compound_cell[cells].std().sort_values(ascending=False)

cell_std_max_min = [cell_std.iloc[[0]].index,

                    cell_std.iloc[[-1]].index]

cell_std_max_min
# plot distribution of sample for cell with MIN variance viability

g = sns.FacetGrid(df_compound_cell, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'c-74')

g.add_legend()
# plot distribution of sample for cell with MAX variance viability

g = sns.FacetGrid(df_compound_cell, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'c-63')

g.add_legend()
# plot correlation matrix between the first 20 cells types

corr = df_compound_cell[cells[0:20]].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(9, 7))

cmap = sns.color_palette('coolwarm')

sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1,

            vmax=1, square=True, cbar_kws={"shrink":.5})
df_compound_gene = df_compound[features_quali + genes]

df_compound_gene.head(3)
gene_std = df_compound_gene[genes].std().sort_values(ascending=False)

gene_std_max_min = [gene_std.iloc[[0]].index,

                    gene_std.iloc[[-1]].index]

gene_std_max_min
# plot distribution of gene with MIN variance viability

g = sns.FacetGrid(df_compound_gene, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'g-219')

g.add_legend()
# plot distribution of gene with MAX variance viability

g = sns.FacetGrid(df_compound_gene, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'g-50')

g.add_legend()
# plot correlation matrix between the first 20 genes expressions

corr = df_compound_gene[genes[0:20]].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(9, 7))

cmap = sns.color_palette('coolwarm')

sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1,

            vmax=1, square=True, cbar_kws={"shrink":.5})
# plot correlation matrix between genes expression and cells viability

corr = df_compound[genes + cells].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(10, 15))

cmap = sns.color_palette('coolwarm')

sns.heatmap(corr.loc[genes, cells], cmap=cmap, center=0, vmin=-1,

            vmax=1, cbar_kws={"shrink":.5})
# plot distribution of sample for gene with MAX variance viability under compound

g = sns.FacetGrid(df_compound_cell, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'c-63')

g.add_legend()
# plot distribution of same cell under control

g = sns.FacetGrid(df_control_cell, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'c-63')

g.add_legend()
# plot distribution of gene with MAX variance viability under compound

g = sns.FacetGrid(df_compound_gene, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'g-50')

g.add_legend()
# plot distribution of same gene under control

g = sns.FacetGrid(df_control_gene, col='cp_time', hue='cp_dose')

g.map(sns.kdeplot, 'g-50')

g.add_legend()
df_moa = df_data[scored_targets + no_scored_targets]

df_moa.head(3)
print('MoA scored count : {}'.format(len(scored_targets)))

print('MoA not scored count : {}'.format(len(no_scored_targets)))
# MoA count in train set (MoA scored)

df_moa_count = df_moa[scored_targets].sum().sort_values(ascending=False)

df_moa_count
# distribution of label occurence in train set (MoA scored)

ax = sns.distplot(df_moa_count, kde=False)

ax.set_xlabel('MoA occurence in dataset')

ax.set_ylabel('Compound count')
# MoA not present in dataset (MoA scored and not scored)

df_moa_count = df_moa.sum().sort_values(ascending=False)

print('MoA not labelled in dataset : {}'.format(df_moa_count[df_moa_count == 0].shape[0]))

no_moa = list(df_moa_count[df_moa_count == 0].index)
# MoA not present in dataset and scored

inter_moa_scored = set(no_moa) & set(scored_targets)

inter_moa_scored
# MoA not present in dataset and not scored

inter_moa_no_scored = set(no_moa) & set(no_scored_targets)

print('MoA not scored and not labellized : {}'.format(len(inter_moa_no_scored)))

print(list(inter_moa_no_scored))
# Number of scored MoA per compound

df_label_count = df_moa[scored_targets].sum(axis=1).sort_values(ascending=False)

df_label_count.describe()
# compound without scored MoA

print('Samples without scored MoA count : {}'.format(

            df_label_count[df_label_count == 0].shape[0]))
# distribution of scored MoA per compound

ax = sns.distplot(df_label_count, kde=False)

ax.set_xlabel('scored MoA in sample')

ax.set_ylabel('compound count')
# plot correlation matrix between scored MoA

corr = df_moa[scored_targets].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(9, 7))

cmap = sns.color_palette('coolwarm')

sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1,

            vmax=1, square=True, cbar_kws={"shrink":.5})
X = df_compound[genes + cells] # no dose and no type

X_embedded = TSNE(n_components=2, init='pca', n_jobs=4).fit_transform(X)

X_embedded.shape
plt.figure(figsize=(15,10))

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=df_compound['nfkb_inhibitor'], alpha=0.2)

plt.title('T-SNE on genes and cells features : nfkb_inhibitor clusters (yellow)')