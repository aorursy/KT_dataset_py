%%capture
# create conda environment with recquired packages
# this takes ~10-15 mins
!conda create -n alcapone_rapids -c rapidsai -c nvidia -c conda-forge rapids scikit-fuzzy python=3.6 cudatoolkit=10.1 -y
# this is to make the conda packages accessible
import sys
sys.path = ["/opt/conda/envs/alcapone_rapids/lib/python3.6/site-packages"]+ sys.path
sys.path = ["/opt/conda/envs/alcapone_rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/alcapone_rapids/lib"] + sys.path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import skfuzzy as fuzz
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance as eudist
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from cuml.manifold import TSNE as cTSNE
from cuml import KMeans as cKMeans
from IPython.display import Image, display
# define global variables
ROOT_PATH = '/kaggle/input/CORD-19-research-challenge/'
METADATA_PATH = f'{ROOT_PATH}/metadata.csv'
# load metadata into a df and look at the contents
meta_df = pd.read_csv(METADATA_PATH, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
def create_embedding_dict(filepath, sample_size=None):
    """create embedding dictionary from file at given filepath"""

    embedding_dict = {}
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            # exit the loop if the desired sample size is reached
            if sample_size and i == sample_size:
                break
            embed = np.zeros((768,))
            for idx, val in enumerate(row):
                if idx > 0:
                    embed[idx-1] = float(val)
            embedding_dict[row[0]] = embed
    return embedding_dict
embedding_dict = create_embedding_dict(f'{ROOT_PATH}/cord19_specter_embeddings_2020-04-10/cord19_specter_embeddings_2020-04-10.csv',
                                       sample_size=None
                                      )
embedding_mat = np.array(list(embedding_dict.values()))
embedding_mat.shape, len(embedding_dict)
n_clusters = 10
def fuzzy_clustering(all_embedding, n_clusters):
    """returns clusters and centroids as results of fuzzy c-means clustering"""
    
    centroids, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=all_embedding.T, 
                                                          c=n_clusters, 
                                                          m=2, 
                                                          error=0.5, 
                                                          maxiter=1000, 
                                                          init=None)
    clusters = np.argmax(u, axis=0)
    return clusters, centroids
def get_clusters(embedding_dict, n_clusters, clusters, centroids = None, k = 5):
    """returns dictionary for clusters"""
    
    cluster_dict = {}
    distance_dict = {}
    for i in range(n_clusters):
        cluster_dict[i] = []
        distance_dict[i] = []
        for j in np.where(clusters == i)[0]:
            paper_id = list(embedding_dict.keys())[j]
            cluster_dict[i].append(paper_id)
            if centroids is not None:
                distance = eudist.euclidean(embedding_mat[j], centroids[i])
                distance_dict[i].append(distance)
    
    if centroids is not None:
        closest_dict = {}
        for i in range(n_clusters):
            closest_idx = np.argsort(distance_dict[i])[0:k]
            closest_dict[i] = []
            for j in range(min(k, len(closest_idx))):
                closest_dict[i].append(cluster_dict[i][j])
        return cluster_dict, closest_dict
    else:
        return cluster_dict
fuzzy_clusters, fuzzy_centroids = fuzzy_clustering(embedding_mat, n_clusters)
fuzzy_clusters_dict, fuzzy_closest_dict = get_clusters(embedding_dict, n_clusters, fuzzy_clusters, fuzzy_centroids)
def get_pca(all_embedding):
    """returns result of pca given an embedding matrix"""
    
    pca = PCA()
    pca_result = pca.fit_transform(all_embedding)
    return pca_result
def plot_pca(pca_result, clusters, title):
    """plots and saves pca result image"""
    
    sns.set(rc={'figure.figsize':(10, 10)})
    palette = sns.color_palette("bright", len(set(clusters)))
    sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=clusters, legend='full', palette=palette)
    
    plt.title(title)
    plt.savefig(f"/kaggle/working/{title}.png")
    plt.show()
fuzzy_pca = get_pca(embedding_mat)
#use PCA to plot embeddings v. fuzzy output clusters
plot_pca(fuzzy_pca, fuzzy_clusters, "PCA Covid-19 Articles - Clustered(Fuzzy C-Means)")
def get_tsne(all_embedding):
    """returns result of TNSE given an embedding matrix"""
    
    tsne = cTSNE(verbose=1)
    tsne_result = tsne.fit_transform(all_embedding)
    return tsne_result
def plot_tsne(tsne_result, clusters, title):
    """plots and saves tsne result image """
    
    sns.set(rc={'figure.figsize':(10, 10)})
    palette = sns.color_palette("bright", len(set(clusters)))
    sns.scatterplot(tsne_result[:,0], tsne_result[:,1], hue=clusters, legend='full', palette=palette)
    
    plt.title(title)
    plt.savefig(f"/kaggle/working/{title}.png")
    plt.show()
fuzzy_tsne = get_tsne(embedding_mat)
#use tSNE to plot embeddings v. fuzzy output clusters 
plot_tsne(fuzzy_tsne, fuzzy_clusters, "t-SNE Covid-19 Articles - Clustered(Fuzzy C-Means)")
for cluster, paper_id in fuzzy_closest_dict.items():
    print(f"Cluster {cluster} - Titles")
    for idx in paper_id:
        print(f"{meta_df['title'].loc[meta_df['cord_uid'] == idx].values[0]}")
def kmeans_clustering(all_embedding, n_clusters):
    """returns result of k-means clustering"""
    
    kmeans = cKMeans(n_clusters=n_clusters, random_state=0).fit(all_embedding)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return clusters, centroids
kmeans_clusters, kmeans_centroids = kmeans_clustering(embedding_mat, n_clusters)
kmeans_clusters_dict, kmeans_closest_dict = get_clusters(embedding_dict, n_clusters, kmeans_clusters, kmeans_centroids)
kmeans_pca = get_pca(embedding_mat)
plot_pca(kmeans_pca, kmeans_clusters, "PCA Covid-19 Articles - Clustered(kmeans)")
kmeans_tsne = get_tsne(embedding_mat)
plot_tsne(kmeans_tsne, kmeans_clusters, "t-SNE Covid-19 Articles - Clustered(kmeans)")

for cluster, paper_id in kmeans_closest_dict.items():
    print(f"Cluster {cluster} - Titles")
    for idx in paper_id:
        print(f"{meta_df['title'].loc[meta_df['cord_uid'] == idx].values[0]}")
# this section contains code we used to generate the clusters using hierarchical clustering
# to run it on kaggle kernel, we suggest limiting the sample size to 1000
# to do so, regenerate the embedding matrix by running the following:

# embedding_dict = create_embedding_dict(f'{ROOT_PATH}/cord19_specter_embeddings_2020-04-10/cord19_specter_embeddings_2020-04-10.csv',
#                                        sample_size=1000,
#                                       )
# embedding_mat = np.array(list(embedding_dict.values()))
# embedding_mat.shape, len(embedding_dict)
def hierarchical_clustering(all_embedding, n_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters).fit(all_embedding)
    clusters = hierarchical.labels_
    return clusters
hierarchical_clusters = hierarchical_clustering(embedding_mat, n_clusters)
hierarchical_clusters_dict = get_clusters(embedding_dict, n_clusters, hierarchical_clusters)
hierarchical_pca = get_pca(embedding_mat)
plot_pca(hierarchical_pca, hierarchical_clusters, "PCA Covid-19 Articles - Clustered(Hierarchical)")
hierarchical_tsne = get_tsne(embedding_mat)
plot_tsne(hierarchical_tsne, hierarchical_clusters, "t-SNE Covid-19 Articles - Clustered(Hierarchical)")
hierarchical_pca = Image("/kaggle/input/results/PCA Covid-19 Articles - Clustered(Hierarchical).png")
hierarchical_tsne = Image("/kaggle/input/results/t-SNE Covid-19 Articles - Clustered(Hierarchical).png")
display(hierarchical_pca, hierarchical_tsne)