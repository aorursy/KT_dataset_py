from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm

import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (20.0, 5.0)
# Read in the survey results, shuffle results
print(f">> Loading data")
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False).sample(frac=1)

# Columns with multiple choice options
MULTIPLE_CHOICE = [
    'CommunicationTools','EducationTypes','SelfTaughtTypes','HackathonReasons', 
    'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith',
    'PlatformDesireNextYear','Methodology','VersionControl',
    'AdBlockerReasons','AdsActions','ErgonomicDevices','Gender',
    'SexualOrientation','RaceEthnicity', 'LanguageWorkedWith',
    'IDE', 'FrameworkWorkedWith', 'FrameworkDesireNextYear',
    'LanguageDesireNextYear', 'DevType',
]

# Columns which we are not interested in
DROP_COLUMNS = [
    'Salary', 'SalaryType', 'Respondent', 'CurrencySymbol'
]

# Drop too easy columns
print(f">> Deleting uninteresting or redundant columns: {DROP_COLUMNS}")
df.drop(DROP_COLUMNS, axis=1, inplace=True)

# Go through all object columns
for c in MULTIPLE_CHOICE:
    
    # Check if there are multiple entries in this column
    temp = df[c].str.split(';', expand=True)

    # Get all the possible values in this column
    new_columns = pd.unique(temp.values.ravel())
    for new_c in new_columns:
        if new_c and new_c is not np.nan:

            # Create new column for each unique column
            idx = df[c].str.contains(new_c, regex=False).fillna(False)
            df.loc[idx, f"{c}_{new_c}"] = 1

    # Info to the user
    print(f">> Multiple entries in {c}. Added {len(new_columns)} one-hot-encoding columns")

    # Drop the original column
    df.drop(c, axis=1, inplace=True)
        
# For all the remaining categorical columns, create dummy columns
df = pd.get_dummies(df)

# Fill in missing values
df.dropna(axis=1, how='all', inplace=True)
dummy_columns = [c for c in df.columns if len(df[c].unique()) == 2]
non_dummy = [c for c in df.columns if c not in dummy_columns]
df[dummy_columns] = df[dummy_columns].fillna(0)
df[non_dummy] = df[non_dummy].fillna(df[non_dummy].median())
print(f">> Filled NaNs in {len(dummy_columns)} OHE columns with 0")
print(f">> Filled NaNs in {len(non_dummy)} non-OHE columns with median values")

# Create correlation matrix
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
print(f">> Dropping the following columns due to high correlations: {to_drop}")
df = df.drop(to_drop, axis=1)

# Perform scaling on all non-dummy columns. Create X and y
nondummy_columns = [c for c in df.columns if df[c].max() > 1]
X = deepcopy(df)
X.loc[:, nondummy_columns] = scale(df[nondummy_columns])
X.drop('ConvertedSalary', axis=1, inplace=True)
print(f">> Shape of final dataframe X: {X.shape}")
# Create a PCA object, specifying how many components we wish to keep
pca = PCA(n_components=50)

# Run PCA on scaled numeric dataframe, and retrieve the projected data
pca_trafo = pca.fit_transform(X)

# The transformed data is in a numpy matrix. This may be inconvenient if we want to further
# process the data, and have a more visual impression of what each column is etc. We therefore
# put transformed/projected data into new dataframe, where we specify column names and index
pca_df = pd.DataFrame(
    pca_trafo,
    index=X.index,
    columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
)
# Plot the explained variance# Plot t 
plt.plot(
    pca.explained_variance_ratio_, "--o", linewidth=2,
    label="Explained variance ratio"
)

# Plot the cumulative explained variance
plt.plot(
    pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
    label="Cumulative explained variance ratio"
)

# Show legend
plt.ylim([-0.1, 0.7])
plt.legend(loc="best", frameon=True)
plt.show()
_, axes = plt.subplots(2, 2, figsize=(15, 10))

dev_types = [c for c in X.columns if 'DevType' in c]
colors = sns.color_palette('hls', len(dev_types)).as_hex()

for i, dev in enumerate(dev_types):   
    idx = (df[dev] == 1)
    pca_df.loc[idx].plot(kind="scatter", x="PC1", y="PC2", ax=axes[0][0], c=colors[i], alpha=0.1)
    pca_df.loc[idx].plot(kind="scatter", x="PC2", y="PC3", ax=axes[0][1], c=colors[i], alpha=0.1, label=dev)
    pca_df.loc[idx].plot(kind="scatter", x="PC3", y="PC4", ax=axes[1][0], c=colors[i], alpha=0.1)
    pca_df.loc[idx].plot(kind="scatter", x="PC4", y="PC5", ax=axes[1][1], c=colors[i], alpha=0.1)
    
axes[0][1].legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show()
# Create a sample dataset
sample = X.sample(30000)
%%time 
print(">> Clustering using HDBSCAN")
clusterer = hdbscan.HDBSCAN(min_cluster_size=500)
clusterer.fit(sample)
%%time
print(">> Dimensionality reduction using TSNE")
projection = TSNE(init='pca', random_state=42).fit_transform(sample)
def get_cluster_colors(clusterer, palette='Paired'):
    """Create cluster colors based on labels and probability assignments"""
    n_clusters = len(np.unique(clusterer.labels_))
    color_palette = sns.color_palette(palette, n_clusters)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    if hasattr(clusterer, 'probabilities_'):
        cluster_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_colors

# Create the plot on the TSNE projection with HDBSCAN colors
_, ax = plt.subplots(1, figsize=(20, 10))
ax.scatter(
    *projection.T, 
    s=50, linewidth=0, 
    c=get_cluster_colors(clusterer), 
    alpha=0.25
)
plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(random_state=42)
skplt.cluster.plot_elbow_curve(kmeans, sample, cluster_ranges=[1, 5, 10, 50, 100, 200])
plt.show()
# Create the plot on the TSNE projection with HDBSCAN colors
_, ax = plt.subplots(1, figsize=(20, 10))
kmeans = KMeans(n_clusters=6).fit(sample)
ax.scatter(
    *projection.T, 
    s=50, linewidth=0, 
    c=get_cluster_colors(kmeans, 'hls'), 
    alpha=0.25
)
plt.show()
perplexities = [5, 30, 50, 100]

_, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, perplexity in tqdm(enumerate(perplexities)):
    
    # Create projection
    projection = TSNE(init='pca', perplexity=perplexity).fit_transform(sample)
    
    # Plot for HDBSCAN clusters
    axes[0, i].set_title("Perplexity=%d" % perplexity)
    axes[0, i].scatter(
        *projection.T, 
        s=50, linewidth=0, 
        c=get_cluster_colors(clusterer), 
        alpha=0.25
    )
    
    # Plot for KMeans clusters
    axes[1, i].scatter(
        *projection.T, 
        s=50, linewidth=0, 
        c=get_cluster_colors(kmeans, 'hls'), 
        alpha=0.25
    )

plt.show()
# Get the data for each cluster (not noise, aka -1)
unique_clusters = [c for c in np.unique(clusterer.labels_) if c > -1]
        
# Create a figure for holding the correlation plots
cols = 2
rows = np.ceil(len(unique_clusters) / cols).astype(int)
_, axes = plt.subplots(rows, cols, figsize=(20, 10*rows))
if rows > 1:
    axes = [x for l in axes for x in l]

# Calculate sample means
sample_mean = sample.median()

# Go through clusters identified by HDBSCAN
for i, label in enumerate(unique_clusters):
    
    # Get index of this cluster
    idx = clusterer.labels_ == label
    
    # Identify feature where the median differs significantly
    median_diff = (sample.median() - sample[idx].median()).abs().sort_values(ascending=False)
    
    # Create boxplot of these features for all vs cluster
    top = median_diff.index[0:20]
    temp_concat = pd.concat([sample.loc[:, top], sample.loc[idx, top]], axis=0).reset_index(drop=True)
    temp_concat['Cluster'] = 'Cluster {}'.format(i+1)
    temp_concat.loc[0:len(sample),'Cluster'] = 'All respondees'
    temp_long = pd.melt(temp_concat, id_vars='Cluster')
    
    sns.boxplot(x='variable', y='value', hue='Cluster', data=temp_long, ax=axes[i])
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(90)
    axes[i].set_title(f'Cluster #{i+1} - {idx.sum()} respondees')    

# Tight layout    
plt.tight_layout()
plt.show()
