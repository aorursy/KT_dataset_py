import numpy as np

import pandas as pd

import scipy.stats as stats

from scipy.spatial.distance import cdist

import collections

import pickle

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
metadata_df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', index_col='cord_uid')
metadata_df
example_df = pd.read_csv('../input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/How does temperature and humidity affect the transmission of 2019-nCoV.csv')
example_df
# find the example sha and uids for the example papers

example_shas = []

example_uids = []

for index, row in example_df.iterrows():

    study_title = row['Study']

    study_metadata = metadata_df[metadata_df['title'] == study_title]

    if len(study_metadata) != 0:

        sha = study_metadata.iloc[0]['sha']

        uid = study_metadata.iloc[0].name

        if str(sha) != 'nan':

            example_shas.append(sha)

            example_uids.append(uid)
example_uids
unique_example_uids = set(example_uids)

len(unique_example_uids)
embeddings_df = pd.read_csv('../input/CORD-19-research-challenge/cord_19_embeddings_4_24/cord_19_embeddings_4_24.csv', header=None, index_col=0)
available_uids = unique_example_uids.intersection(embeddings_df.index) # select example uids with an available embedding

example_embeddings_df = embeddings_df.loc[available_uids]
example_embeddings_df
# first lets see some plots of the embeddings features for the examples vs the rest of the papers

for i in range(1, 21, 2):

    plt.scatter(embeddings_df[i], embeddings_df[i+1])

    plt.scatter(example_embeddings_df[i], example_embeddings_df[i+1])

    plt.show()
# First, lets get the population mean for each embedding feature

feature_pop_means = embeddings_df.mean(0)
# Now run the t-tests

t_stats, p_vals = stats.ttest_1samp(example_embeddings_df, feature_pop_means)
# here we show some visualizations of feature significance

plt.bar(range(len(p_vals)), -np.log(p_vals)) # we plot the negative log of the p-values for ease of visualization purposes

plt.hlines(-np.log(0.05), 0, 800) # line representing p-value of 0.05

plt.hlines(-np.log(0.05/len(p_vals)), 0, 800) # line representing a bonferroni adjusted p-value cutoff
# select the subset of informative features from original dataframe (selecting p-values < 0.05/len(p_vals))

informative_embeddings_df = embeddings_df.loc[:, p_vals < 0.05/len(p_vals)]
informative_embeddings_df
clustering = KMeans(n_clusters=10, random_state=0).fit(informative_embeddings_df.values)
# get the cluster labels

labels = clustering.labels_
collections.Counter(labels)
uid_cluster_map = dict(zip(informative_embeddings_df.index, labels)) # map the uids to the cluster labels
example_clusters = [uid_cluster_map[uid] for uid in example_embeddings_df.index] # find the cluster assigned to each of the example papers
# looks like cluster 4 contains all the papers from our examples.

example_clusters
for i in range(1, 11):

    print('cluster', i)

    cluster_ids = set([k for k, v in uid_cluster_map.items() if v == i])

    cluster_ids = cluster_ids.intersection(metadata_df.index)

    for ele in metadata_df.loc[cluster_ids, 'title']:

        if isinstance(ele, str):

            # use this rule to filter out non-relevant papers and focus on the coronavirus related content

            if ('corona' in ele.lower() or 'cov' in ele.lower()) and ('humid' in ele.lower() or 'temperature' in ele.lower()):

                print('\t', ele)

    print('#'*25)