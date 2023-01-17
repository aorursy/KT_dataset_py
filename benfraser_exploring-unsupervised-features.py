import matplotlib.pyplot as plt

import pandas as pd

import os

import numpy as np

import seaborn as sns



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans, DBSCAN

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.manifold import TSNE

from sklearn.metrics import log_loss, silhouette_score

from sklearn.mixture import GaussianMixture

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_validate, cross_val_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler



from tqdm import tqdm
input_dir = '/kaggle/input/lish-moa'

train_features = pd.read_csv(os.path.join(input_dir, 'train_features.csv'))

train_targets_scored = pd.read_csv(os.path.join(input_dir, 'train_targets_scored.csv'))

train_targets_nonscored = pd.read_csv(os.path.join(input_dir, 'train_targets_nonscored.csv'))

test_features = pd.read_csv(os.path.join(input_dir, 'test_features.csv'))
train_features.shape, train_targets_scored.shape, train_targets_nonscored.shape, test_features.shape
cat_cols = ['cp_type', 'cp_time', 'cp_dose']



plt.figure(figsize=(16,4))



for idx, col in enumerate(cat_cols):

    plt.subplot(int(f'13{idx + 1}'))

    labels = train_features[col].value_counts().index.values

    vals = train_features[col].value_counts().values

    sns.barplot(x=labels, y=vals)

    plt.xlabel(f'{col}')

    plt.ylabel('Count')

plt.tight_layout()

plt.show()
# select all indices when 'cp_type' is 'ctl_vehicle'

ctl_vehicle_idx = (train_features['cp_type'] == 'ctl_vehicle')



# evaluate number of 1s we have in the total train scores when cp_type = ctl_vehicle

train_targets_scored.loc[ctl_vehicle_idx].iloc[:, 1:].sum().sum()
# take a copy of all our training sig_ids for reference

train_sig_ids = train_features['sig_id'].copy()
# drop cp_type column since we no longer need it

X = train_features.drop(['sig_id', 'cp_type'], axis=1).copy()

X = X.loc[~ctl_vehicle_idx].copy()



y = train_targets_scored.drop('sig_id', axis=1).copy()

y = y.loc[~ctl_vehicle_idx].copy()



X.shape, y.shape
X.head(3)
cat_feats = X.iloc[:, :2].copy()

X_cell_v = X.iloc[:, -100:].copy()

X_gene_e = X.iloc[:, 2:772].copy()
cat_feats.head(3)
X_cell_v.head(3)
X_gene_e.head(3)
sns.distplot(X_cell_v)

plt.show()
sns.distplot(X_gene_e)

plt.show()
cat_feats = X.iloc[:, :2].copy()

X_cell_v = X.iloc[:, -100:].copy()

X_gene_e = X.iloc[:, 2:772].copy()
def plot_features(X, y, selected_idx, features_type, figsize=(14,10)):

    x_range = range(1, X.shape[1] + 1)

    

    fig = plt.figure(figsize=(14,10))

    

    for i, idx in enumerate(selected_idx):

        ax = fig.add_subplot(selected_idx.shape[0], 1, i + 1)

        vals = X.iloc[idx].values

    

        if (y.iloc[idx] == 1).sum():

            output_labels = list(y.iloc[idx][y.iloc[idx] == 1].index.values)

        

            labels = " ".join(output_labels)

        else:

            labels = "None (all labels zero)"

        

        sns.lineplot(x_range, vals)

        plt.title(f"Row {idx}, Labels: {labels}", weight='bold')

        plt.xlim(0.0, X.shape[1])

        plt.grid()



    plt.xlabel(f"{features_type}", weight='bold', size=14)

    plt.tight_layout()

    plt.show()

    

    

def plot_mean_std(dataframe, feature_name, features_type, figsize=(14,6), alpha=0.3):

    """ Plot rolling mean and standard deviation for given dataframe """

    

    plt.figure(figsize=figsize)

    

    x_range = range(1, dataframe.shape[1] + 1)

    

    chosen_rows = y.loc[y[feature_name] == 1]

    chosen_feats = dataframe.loc[y[feature_name] == 1]

    

    means = chosen_feats.mean()

    stds = chosen_feats.std()

    

    plt.plot(x_range, means, label=feature_name)    

    plt.fill_between(x_range, means - stds, means + stds, 

                         alpha=alpha)



    plt.title(f'{features_type}: {feature_name} - Mean & Standard Deviation', weight='bold')

    

    plt.xlim(0.0, dataframe.shape[1])

    

    plt.show()
# lets plot some random rows from our data

random_idx = np.random.randint(X.shape[0], size=(5,))



plot_features(X_cell_v, y, random_idx, features_type='Cell Features')
plot_features(X_gene_e, y, random_idx, features_type='Gene Features')
# select an output label to plot associated training features

chosen_label = 'btk_inhibitor'

chosen_rows = y.loc[y[chosen_label] == 1]

chosen_feats = X_gene_e.loc[y[chosen_label] == 1]



# select random rows from those available above for the chosen label

random_idx = np.random.choice(range(0, chosen_rows.shape[0]), size=(5,), replace=False)
plot_features(chosen_feats, chosen_rows, random_idx, features_type='Gene Features')
plot_mean_std(X_gene_e, 'btk_inhibitor', 'Gene Features')
# select an output label to plot associated training features

chosen_label = 'histamine_receptor_antagonist'

chosen_rows = y.loc[y[chosen_label] == 1]

chosen_feats = X_gene_e.loc[y[chosen_label] == 1]



# select random rows from those available above for the chosen label

random_idx = np.random.choice(range(0, chosen_rows.shape[0]), size=(5,))



plot_features(chosen_feats, chosen_rows, random_idx, features_type='Gene Features')
plot_mean_std(X_gene_e, 'histamine_receptor_antagonist', 'Gene Features')
# select an output label to plot associated training features

chosen_label = 'free_radical_scavenger'

chosen_rows = y.loc[y[chosen_label] == 1]

chosen_feats = X_gene_e.loc[y[chosen_label] == 1]



# select random rows from those available above for the chosen label

random_idx = np.random.choice(range(0, chosen_rows.shape[0]), size=(5,))



plot_features(chosen_feats, chosen_rows, random_idx, features_type='Gene Features')
plot_mean_std(X_gene_e, 'free_radical_scavenger', 'Gene Features')
X_sample = X.sample(10000, random_state=12)

X_cell_v = X_sample.iloc[:, -100:].copy()

X_gene_e = X_sample.iloc[:, 2:772].copy()

X_cell_gene = X_sample.iloc[:, 2:].copy()
k_range = [x for x in range(1, 25, 1)]
%time k_kmeans = [KMeans(n_clusters=k, random_state=12).fit(X_cell_v) for k in k_range]

inertias = [model.inertia_ for model in k_kmeans]
plt.figure(figsize=(12, 5))

sns.lineplot(k_range, inertias)

sns.scatterplot(k_range, inertias)

plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Inertia", fontsize=14, weight='bold')

plt.grid()

plt.xlim(0.0, 15.0)

plt.show()
%time silhouette_scores = [silhouette_score(X_cell_v, model.labels_) for model in k_kmeans[1:]]
plt.figure(figsize=(14, 6))

sns.lineplot(k_range[1:], silhouette_scores)

sns.scatterplot(k_range[1:], silhouette_scores)

plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Silhoutte Score", fontsize=14, weight='bold')

plt.grid()

plt.xlim(0.0, 15.0)

plt.show()
cell_k = 10

kmeans = KMeans(n_clusters=cell_k)

km_cell_feats = kmeans.fit_transform(X_cell_v)

kmeans_cell_labels = kmeans.predict(X_cell_v)



km_cell_feats.shape, kmeans_cell_labels.shape
tsne = TSNE(verbose=1, perplexity=100, n_jobs=-1)

%time X_cell_embedded = tsne.fit_transform(km_cell_feats)
# sns settings

sns.set(rc={'figure.figsize':(14,10)})

palette = sns.hls_palette(cell_k, l=.4, s=.8)



# plot t-SNE with annotations from k-means clustering

sns.scatterplot(X_cell_embedded[:,0], X_cell_embedded[:,1], 

                hue=kmeans_cell_labels, legend='full', palette=palette)

plt.title('t-SNE on our Cell data with K-Means Clustered labels', weight='bold')

plt.show()
k_range = [x for x in range(1, 10)]

k_range.extend([x for x in range(10, 21, 2)])

aic_scores = []

bic_scores = []



for k in tqdm(k_range):

    gm_k = GaussianMixture(n_components=k, n_init=10, random_state=12).fit(X_cell_v)

    aic_scores.append(gm_k.aic(X_cell_v))

    bic_scores.append(gm_k.bic(X_cell_v))
plt.figure(figsize=(12, 5))

sns.lineplot(k_range, aic_scores, color="tab:blue", label='AIC')

sns.scatterplot(k_range, aic_scores, color="tab:blue")



sns.lineplot(k_range, bic_scores, color="tab:green", label='BIC')

sns.scatterplot(k_range, bic_scores, color="tab:blue")



plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Information Criterion", fontsize=14, weight='bold')

plt.legend()

plt.grid()

plt.show()
print(f"AIC minimum at {k_range[np.argmin(aic_scores)]} clusters.")

print(f"BIC minimum at {k_range[np.argmin(bic_scores)]} clusters.")
k_range = [x for x in range(1, 25, 1)]

k_range.extend([50, 100, 150, 200, 250])
%time k_kmeans = [KMeans(n_clusters=k, random_state=12).fit(X_gene_e) for k in k_range]

inertias = [model.inertia_ for model in k_kmeans]
plt.figure(figsize=(12, 5))

sns.lineplot(k_range, inertias)

sns.scatterplot(k_range, inertias)

plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Inertia", fontsize=14, weight='bold')

plt.grid()

plt.xlim(0.0, 25.0)

plt.show()
%time silhouette_scores = [silhouette_score(X_gene_e, model.labels_) for model in k_kmeans[1:]]
plt.figure(figsize=(14, 6))

sns.lineplot(k_range[1:], silhouette_scores)

sns.scatterplot(k_range[1:], silhouette_scores)

plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Silhoutte Score", fontsize=14, weight='bold')

plt.grid()

plt.xlim(0.0, 15.0)

plt.show()
gene_k = 6

kmeans = KMeans(n_clusters=gene_k)

km_gene_feats = kmeans.fit_transform(X_gene_e)

kmeans_gene_labels = kmeans.predict(X_gene_e)



km_gene_feats.shape, kmeans_gene_labels.shape
tsne = TSNE(verbose=1, perplexity=100, n_jobs=-1)

%time X_gene_embedded = tsne.fit_transform(km_gene_feats)
# sns settings

sns.set(rc={'figure.figsize':(14,10)})

palette = sns.hls_palette(gene_k, l=.4, s=.8)



# plot t-SNE with annotations from k-means clustering

sns.scatterplot(X_gene_embedded[:,0], X_gene_embedded[:,1], 

                hue=kmeans_gene_labels, legend='full', palette=palette)

plt.title('t-SNE with labels obtained from K-Means Clustering', weight='bold')

plt.show()
pca_tf_gene = PCA(n_components=0.90)

X_gene_e_red = pca_tf_gene.fit_transform(X_gene_e)

print(f"Original data: {X_gene_e.shape} \nPCA Reduced data: {X_gene_e_red.shape}")
k_range = [x for x in range(1, 11)]

k_range.extend([12, 15, 30, 50])

gene_aic_scores = []

gene_bic_scores = []



for k in tqdm(k_range):

    gm_k = GaussianMixture(n_components=k, n_init=10, random_state=12).fit(X_gene_e_red)

    gene_aic_scores.append(gm_k.aic(X_gene_e_red))

    gene_bic_scores.append(gm_k.bic(X_gene_e_red))
plt.figure(figsize=(12, 5))

sns.lineplot(k_range, gene_aic_scores, color="tab:blue", label='AIC')

sns.scatterplot(k_range, gene_aic_scores, color="tab:blue")



sns.lineplot(k_range, gene_bic_scores, color="tab:green", label='BIC')

sns.scatterplot(k_range, gene_bic_scores, color="tab:blue")



plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Information Criterion", fontsize=14, weight='bold')

plt.title("Gene Features Gaussian Mixture Model Clustering")

plt.legend()

plt.grid()

plt.show()
aic_arr = np.array(gene_aic_scores)

bic_arr = np.array(gene_bic_scores)

total = aic_arr + bic_arr

print(f"Cluster number with minimum sum of AIC and BIC: {k_range[np.argmin(total)]}")
k_range = [x for x in range(1, 25, 1)]

k_range.extend([50, 100, 150, 200, 250])
%time k_kmeans = [KMeans(n_clusters=k, random_state=12).fit(X_cell_gene) for k in k_range]

inertias = [model.inertia_ for model in k_kmeans]
plt.figure(figsize=(12, 6))

sns.lineplot(k_range, inertias)

sns.scatterplot(k_range, inertias)

plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Inertia", fontsize=14, weight='bold')

plt.grid()

plt.xlim(0.0, 50.0)

plt.show()
%time silhouette_scores = [silhouette_score(X_cell_gene, model.labels_) for model in k_kmeans[1:]]
plt.figure(figsize=(14, 6))

sns.lineplot(k_range[1:], silhouette_scores)

sns.scatterplot(k_range[1:], silhouette_scores)

plt.xlabel("Clusters, $k$", fontsize=14, weight='bold')

plt.ylabel("Silhoutte Score", fontsize=14, weight='bold')

plt.grid()

plt.xlim(0.0, 15.0)

plt.show()
combined_k = 4

kmeans = KMeans(n_clusters=combined_k)

km_comb_feats = kmeans.fit_transform(X_cell_gene)

kmeans_comb_labels = kmeans.predict(X_cell_gene)
tsne = TSNE(verbose=1, perplexity=100, n_jobs=-1)

%time X_comb_embedded = tsne.fit_transform(km_comb_feats)
# sns settings

sns.set(rc={'figure.figsize':(14,10)})

palette = sns.hls_palette(combined_k, l=.4, s=.8)



# plot t-SNE with annotations from k-means clustering

sns.scatterplot(X_comb_embedded[:,0], X_comb_embedded[:,1], 

                hue=kmeans_comb_labels, legend='full', palette=palette)

plt.title('t-SNE with labels obtained from K-Means Clustering', weight='bold')

plt.show()
pca_gene = PCA(n_components=0.99)

pca_combined = PCA(n_components=0.99)



X_gene_e_rd = pca_gene.fit_transform(X_gene_e)

X_cell_gene_rd = pca_combined.fit_transform(X_cell_gene)



X_gene_e_rd.shape, X_cell_gene_rd.shape
tsne_cell = TSNE(verbose=1, perplexity=100, n_jobs=-1)

tnse_gene = TSNE(verbose=1, perplexity=100, n_jobs=-1)

tnse_combined = TSNE(verbose=1, perplexity=100, n_jobs=-1)
%time X_cell_v_tsne = tsne_cell.fit_transform(X_cell_v)
%time X_gene_e_tsne = tnse_gene.fit_transform(X_gene_e_rd)
%time X_cell_gene_tsne = tnse_gene.fit_transform(X_gene_e_rd)
fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(1, 3, 1)

sns.scatterplot(X_cell_v_tsne[:,0], X_cell_v_tsne[:,1], legend='full')

ax.set_title('Cell Features t-SNE', weight='bold')



ax = fig.add_subplot(1, 3, 2)

sns.scatterplot(X_gene_e_tsne[:,0], X_gene_e_tsne[:,1], legend='full', color='tab:orange')

ax.set_title('Gene Features t-SNE', weight='bold')



ax = fig.add_subplot(1, 3, 3)

sns.scatterplot(X_cell_gene_tsne[:,0], X_cell_gene_tsne[:,1], legend='full', color='tab:red')

ax.set_title('Combined Gene and Cell Features t-SNE', weight='bold')

plt.show()
pre_dbs_gene_pca = PCA(n_components=10)

pre_dbs_cell_pca = PCA(n_components=10)

pre_dbs_comb_pca = PCA(n_components=10)



cell_reduced = pre_dbs_gene_pca.fit_transform(X_gene_e)

gene_reduced = pre_dbs_cell_pca.fit_transform(X_cell_v)

combined_reduced = pre_dbs_comb_pca.fit_transform(X_cell_gene)
dbscan_cell = DBSCAN(eps=13, min_samples=5)

dbscan_cell.fit(cell_reduced)

np.unique(dbscan_cell.labels_, return_counts=True)
dbscan_gene = DBSCAN(eps=3, min_samples=4)

dbscan_gene.fit(gene_reduced)

np.unique(dbscan_gene.labels_, return_counts=True)
dbscan_comb = DBSCAN(eps=3, min_samples=5)

dbscan_comb.fit(combined_reduced)

np.unique(dbscan_comb.labels_, return_counts=True)
fig = plt.figure(figsize=(17,6))

cell_palette = sns.hls_palette(len(np.unique(dbscan_cell.labels_)), l=.4, s=.8)

ax = fig.add_subplot(1, 3, 1)

sns.scatterplot(X_cell_v_tsne[:,0], X_cell_v_tsne[:,1], 

                hue=dbscan_cell.labels_, legend='full', palette=cell_palette)

ax.set_title('Cell t-SNE & DBSCAN Clusters', weight='bold')



ax = fig.add_subplot(1, 3, 2)

gene_palette = sns.hls_palette(len(np.unique(dbscan_gene.labels_)), l=.4, s=.8)

sns.scatterplot(X_gene_e_tsne[:,0], X_gene_e_tsne[:,1], color='tab:orange',

                hue=dbscan_gene.labels_, legend='full', palette=gene_palette)

ax.set_title('Gene t-SNE & DBSCAN Clusters', weight='bold')



ax = fig.add_subplot(1, 3, 3)

comb_palette = sns.hls_palette(len(np.unique(dbscan_comb.labels_)), l=.4, s=.8)

sns.scatterplot(X_cell_gene_tsne[:,0], X_cell_gene_tsne[:,1], color='tab:red',

                hue=dbscan_comb.labels_, legend='full', palette=comb_palette)

ax.set_title('Combined Gene and Cell t-SNE & DBSCAN Clusters', weight='bold')

plt.tight_layout()

plt.show()
# standardise our numerical features data prior to clustering

std_scaler = StandardScaler()

X.iloc[:, 2:] = std_scaler.fit_transform(X.iloc[:, 2:].values)
cell_kmeans = KMeans(n_clusters=4)

gene_kmeans = KMeans(n_clusters=4)

comb_kmeans = KMeans(n_clusters=4)
# one hot encode our categorical features

X_cats = X.iloc[:, :2].copy()

X_cats['cp_time'] = X_cats['cp_time'].astype('object')

X_cats = pd.get_dummies(X_cats)



# obtain our splits for gene and cell data

X_cell_gene = X.iloc[:, 2:].copy()

X_cell = X.iloc[:, -100:].copy()

X_gene = X.iloc[:, 2:772].copy()



X_cats.shape, X_cell_gene.shape, X_cell_v.shape, X_gene_e.shape
%time X_cell_rd = cell_kmeans.fit_transform(X_cell)
%time X_gene_rd = gene_kmeans.fit_transform(X_gene)
%time X_cell_gene_rd = comb_kmeans.fit_transform(X_cell_gene)
X_cell_rd.shape, X_gene_rd.shape, X_cell_gene_rd.shape
# combine all of our features into one

cat_feats = list(X_cats.columns.values)

cell_feats = [f"cell_clust_{x}" for x in range(1, X_cell_rd.shape[1] + 1)]

gene_feats = [f"gene_clust_{x}" for x in range(1, X_gene_rd.shape[1] + 1)]

combined_feats = [f"cell_gene_clust_{x}" for x in range(1, X_cell_gene_rd.shape[1] + 1)]



combined = np.c_[X_cats, X_cell_rd, X_gene_rd, X_cell_gene_rd]

X_all_rd = pd.DataFrame(combined, columns=cat_feats + cell_feats + gene_feats + combined_feats)

X_all_rd.head(3)
original = np.c_[X_cats, X_cell, X_gene]

X_original = pd.DataFrame(original, columns= cat_feats + 

                          list(X_cell.columns.values) + 

                          list(X_gene.columns.values))

X_original.shape
# evaluate using cross-validation

lin_reg = LinearRegression()

lr_val_preds_0 = cross_val_predict(lin_reg, X_original, y, cv=5)



# in order to effective work out log loss, we need to flatten both arrays before computing log loss

lr_log_loss = log_loss(np.ravel(y), np.ravel(lr_val_preds_0))

print(f"Log loss for our Linear Regression Model: {lr_log_loss:.5f}\n")
# evaluate using cross-validation

lin_reg = LinearRegression()

lr_val_preds_1 = cross_val_predict(lin_reg, X_all_rd, y, cv=5)



# in order to effective work out log loss, we need to flatten both arrays before computing log loss

lr_log_loss = log_loss(np.ravel(y), np.ravel(lr_val_preds_1))

print(f"Log loss for our Linear Regression Model: {lr_log_loss:.5f}\n")
lr_model_1 = LinearRegression().fit(X_all_rd, y)
#et_clf = ExtraTreesClassifier(n_jobs=-1)

#%time et_val_preds = cross_val_predict(et_clf, X_all_rd, y, cv=3)
# in order to effective work out log loss, we need to flatten both arrays before computing log loss

#et_log_loss = log_loss(np.ravel(y), np.ravel(et_val_preds))

#print(f"Log loss for Extra Trees Classifier: {et_log_loss:.5f}\n")
all_combined = np.c_[X_cats, X_cell, X_gene, X_cell_rd, X_gene_rd]

X_extended = pd.DataFrame(all_combined, columns=(cat_feats + list(X_cell.columns.values) +

                                                 list(X_gene.columns.values)+ cell_feats + gene_feats))



X_extended.shape
X_extended.head(3)
# evaluate using cross-validation

lin_reg = LinearRegression()

lr_val_preds_2 = cross_val_predict(lin_reg, X_extended, y, cv=5)



# in order to effective work out log loss, we need to flatten both arrays before computing log loss

lr_log_loss = log_loss(np.ravel(y), np.ravel(lr_val_preds_2))

print(f"Log loss for our Linear Regression Model: {lr_log_loss:.5f}\n")
clustered = np.c_[X_cats, X_cell_rd, X_gene_rd]

X_clustered = pd.DataFrame(clustered, columns= cat_feats + cell_feats + gene_feats)



X_clustered.shape
# evaluate using cross-validation

lin_reg = LinearRegression()

lr_val_preds_3 = cross_val_predict(lin_reg, X_clustered, y, cv=5)



# in order to effective work out log loss, we need to flatten both arrays before computing log loss

lr_log_loss = log_loss(np.ravel(y), np.ravel(lr_val_preds_3))

print(f"Log loss for our Linear Regression Model: {lr_log_loss:.5f}\n")
lr_model_2 = LinearRegression().fit(X_clustered, y)
clustered = np.c_[X_cats, X_cell_gene_rd]

X_clustered = pd.DataFrame(clustered, columns= cat_feats + combined_feats)



X_clustered.shape
# evaluate using cross-validation

lin_reg = LinearRegression()

lr_val_preds_4 = cross_val_predict(lin_reg, X_clustered, y, cv=5)



# in order to effective work out log loss, we need to flatten both arrays before computing log loss

lr_log_loss = log_loss(np.ravel(y), np.ravel(lr_val_preds_4))

print(f"Log loss for our Linear Regression Model: {lr_log_loss:.5f}\n")
lr_model_3 = LinearRegression().fit(X_clustered, y)
avg_val_preds = (lr_val_preds_1 + lr_val_preds_3 + lr_val_preds_4) / 3.0
# in order to effective work out log loss, we need to flatten both arrays before computing log loss

comb_log_loss = log_loss(np.ravel(y), np.ravel(avg_val_preds))

print(f"Log loss for our Linear Regression Model: {comb_log_loss:.5f}\n")
# take a copy of all our training sig_ids for reference

test_sig_ids = test_features['sig_id'].copy()



# select all indices when 'cp_type' is 'ctl_vehicle'

test_ctl_vehicle_idx = (test_features['cp_type'] == 'ctl_vehicle')
X_test = test_features.drop(['sig_id', 'cp_type'], axis=1).copy()



# standardise our test set numerical features

X_test.iloc[:, 2:] = std_scaler.fit_transform(X_test.iloc[:, 2:].values)
X_test_cat = X_test.iloc[:, :2].copy()

X_test_cat['cp_time'] = X_test_cat['cp_time'].astype('object')

X_test_cat = pd.get_dummies(X_test_cat)



X_test_cell = X_test.iloc[:, -100:].copy()

X_test_gene = X_test.iloc[:, 2:772].copy()

X_test_cell_gene = X_test.iloc[:, 2:].copy()



X_test_cat.shape, X_test_cell.shape, X_test_gene.shape, X_test_cell_gene.shape
X_test_cell_rd = cell_kmeans.transform(X_test_cell)

X_test_gene_rd = gene_kmeans.transform(X_test_gene)

X_test_cell_gene_rd = comb_kmeans.transform(X_test_cell_gene)
# combine all of our features into one

test_combined = np.c_[X_test_cat, X_test_cell_rd, X_test_gene_rd, X_test_cell_gene_rd]

X_test_1 = pd.DataFrame(test_combined, columns=cat_feats + cell_feats + gene_feats + combined_feats)



# make predicts on this data using model 1 (trained previously)

model_1_preds = lr_model_1.predict(X_test_1)
test_clustered = np.c_[X_test_cat, X_test_cell_rd, X_test_gene_rd]

X_test_2 = pd.DataFrame(test_clustered, columns= cat_feats + cell_feats + gene_feats)



# make predicts on this data using model 2 (trained previously)

model_2_preds = lr_model_2.predict(X_test_2)
test_clust_comb = np.c_[X_test_cat, X_test_cell_gene_rd]

X_test_3 = pd.DataFrame(test_clust_comb, columns= cat_feats + combined_feats)



# make predicts on this data using model 3 (trained previously)

model_3_preds = lr_model_3.predict(X_test_3)
test_preds = (model_1_preds + model_2_preds + model_3_preds) / 3.0

test_preds.shape
# change all cp_type == ctl_vehicle predictions to zero

test_preds[test_sig_ids[test_ctl_vehicle_idx].index.values] = 0



# confirm all values now sum to zero for these instances

test_preds[test_sig_ids[test_ctl_vehicle_idx].index.values].sum()
# we have some values above 1 and below 0 - this needs amending since probs should only be 0-1

test_preds[test_preds > 1.0] = 1.0

test_preds[test_preds < 0.0] = 0.0



# confirm these values are all corrected

test_preds.max(), test_preds.min()
test_preds = pd.DataFrame(test_preds, columns=train_targets_scored.columns.values[1:])

test_submission = pd.DataFrame({'sig_id' : test_sig_ids})

test_submission[test_preds.columns] = test_preds

test_submission.head(3)
# save our submission as csv

test_submission.to_csv('submission.csv', index=False)