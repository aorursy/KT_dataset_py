# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk # Preprocessing and traditional models

import tensorflow as tf # Building ANNs



# Setting up visualization

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv("/kaggle/input/ess9-preprocessed-data/ESS_values.csv")
# List of 10 Schwarz values 

values = ['conformity', 'tradition', 'benevolence', 'universalism', 'self_direction', 

          'stimulation', 'hedonism', 'achievement', 'power', 'security']
# Plot the distribution of answers for 10 values with indices from 0 to 20. 

def plot_value_freq(val_index = 0):

    plt.figure(figsize = (7, 5))

    plt.title(values[val_index])

    sns.distplot(Data[values[val_index]],  hist=True, rug=False)

    plt.ylabel("approximated density")

    plt.xlabel(values[val_index] + ": individually centered score")
for i in range(0, 10):

    plot_value_freq(i)
# Explore how much linearly independent values actually are using PCA



from sklearn.decomposition import PCA

pca = PCA()

pca.fit(Data[values]);
# Visualizing principal components

plt.figure(figsize = (14, 10))

plt.title("Fraction of information explained by each new principal component in V-space\n(blue line shows a zero covariation case)")

sns.barplot(x = pd.Series(pca.explained_variance_ratio_).index, 

y = pd.Series(pca.explained_variance_ratio_).values)

plt.ylabel("Information explained")

plt.xlabel("Principal component")



# Adding level of information explained when there is absolutely no covariation between factors

no_covariation_expl_inf = 1 / len(values)

plt.axhline(y=no_covariation_expl_inf, linewidth=0.5, color='b', linestyle='--');
# Multiplying design and population weights to calculate total weight

Data['total_weight'] = Data['dweight'] * Data['pweight']
from sklearn.cluster import KMeans



def apply_kmeans(data, sample_weight = None, min_clusters = 2, max_clusters = 10):

    """

    Input is data, min_clusters, max_clusters.

    Data shouldn't contain missing values.

    min_clusters should be integer >= 1.

    max_clusters should be an integer >= min_clusters

    

    Returns a dictionary of k-means models which was fitted to data

    """

    kmeans_models = []

    for n_clusters in range(min_clusters, max_clusters + 1):

        kmeans = KMeans(n_clusters = n_clusters, max_iter = 1000) 

        kmeans.fit(data, sample_weight = sample_weight)

        kmeans_models.append(kmeans)

    return kmeans_models



def parse_kmeans(kmeans):

    """

    Input: already fitted kmeans model

    Output: dictionary which contains most important information of the model: 

    {

    "clusters_num": int

    "observations_clusters": pd.Series

    "clusters_freq": pd.Series

    "cluster_centers": pd.DataFrame (rows are factor axises, columns are clusters)

    "inertia": numpy.float64

    "iterations": int

    }

    """

    parsed_kmeans = {}

    parsed_kmeans["clusters_num"] = len(kmeans.cluster_centers_)

    parsed_kmeans["observations_clusters"] = pd.Series(kmeans.labels_)

    parsed_kmeans["clusters_freq"] = pd.Series(kmeans.labels_).value_counts()

    parsed_kmeans["cluster_centers"] = pd.DataFrame(kmeans.cluster_centers_.transpose())

    parsed_kmeans["inertia"] = kmeans.inertia_

    parsed_kmeans["iterations"] = kmeans.n_iter_

    return parsed_kmeans
kmeans_models = apply_kmeans(data = Data[values], sample_weight = Data["total_weight"], min_clusters = 1, max_clusters = 20);
parsed_kmeans_models = [parse_kmeans(model) for model in kmeans_models];
clusters_number = []

inertias = []

iterations = []

cluster_freq = []

clusters_labels = []

cluster_centers = []



for i in range(0, len(parsed_kmeans_models)):

    model = parsed_kmeans_models[i]

    iterations.append(model["iterations"])

    clusters_number.append(model["clusters_num"])

    inertias.append(model["inertia"])

    cluster_freq.append(model["clusters_freq"]) 

    clusters_labels.append(model["observations_clusters"])

    cluster_centers.append(model["cluster_centers"])

    

clusters_inertias = pd.DataFrame({"clusters": clusters_number, "inertia": inertias})

clusters_iterations = pd.DataFrame({"clusters": clusters_number, "iterations": iterations});
sns.lineplot(x = "clusters", y = "iterations", data = clusters_iterations);



# All clusters have converged since iterations number of each model is less than 1000
plt.figure(figsize = (15,10))

plt.title("Total inertia of observations depending on number of clusters")

sns.lineplot(x = "clusters", y = "inertia", data = clusters_inertias);

clusters_inertias.iloc[0:11]
inertia_drops = []

for i in range(0, len(inertias) - 1):

    diff = abs(inertias[i + 1] - inertias[i])

    inertia_drops.append(diff)

    

clusters_inertia_drops = pd.DataFrame({"cluster": clusters_number[1:], "inertia_drop": inertia_drops});
plt.figure(figsize = (15,10))

plt.title("Inertia drop when adding new cluster")

sns.barplot(x = clusters_inertia_drops["cluster"], y = clusters_inertia_drops["inertia_drop"]);
from sklearn.metrics import silhouette_score



silhouette_scores = []



# Calculating silhouette scores for 2 to 10 clusters

for labels in clusters_labels[1:10]:

    sil_score = silhouette_score(X = Data[values], labels = labels, metric = 'euclidean')

    silhouette_scores.append(sil_score)



# Creating data frame cluster numbers and sihouette scores from 2 clusters (index = 1)

clusters_sil_scores = pd.DataFrame({"clusters": clusters_number[1:10], "silhouette_scores": silhouette_scores});
plt.figure(figsize = (15,10))

plt.title("Silhouette scores for different number of clusters")

sns.lineplot(x = "clusters", y = "silhouette_scores", data = clusters_sil_scores);
clusters_sil_scores.iloc[0:10]
for i in range(1, 7):

    plt.title("Cluster fullness")

    sns.barplot(x = cluster_freq[i].keys().values, y = cluster_freq[i].values)

    plt.xlabel("Cluster index")

    plt.ylabel("Observations number")

    plt.show()
# Saving cluster centers in new variables and setting values variables names as rows



cluster_centers_3 = cluster_centers[2].copy()

cluster_centers_4 = cluster_centers[3].copy()

cluster_centers_5 = cluster_centers[4].copy()

for i in range(0, len(values)):

    cluster_centers_3.rename(index = {i: values[i]}, inplace = True)

    cluster_centers_4.rename(index = {i: values[i]}, inplace = True)

    cluster_centers_5.rename(index = {i: values[i]}, inplace = True)



# Saving cluster labels in new variables

cluster_labels_3 = pd.DataFrame(clusters_labels[2]).rename(columns = {0: "cluster"})

cluster_labels_4 = pd.DataFrame(clusters_labels[3]).rename(columns = {0: "cluster"})

cluster_labels_5 = pd.DataFrame(clusters_labels[4]).rename(columns = {0: "cluster"})
def visualize_clusters_center(cluster_centers_data):

    plt.figure(figsize=(10, 10))

    plt.title("Cluster centers coordinates in V-space")

    sns.heatmap(data=cluster_centers_data, annot=True)

    plt.xlabel("Cluster center")

    plt.ylabel("Values")

    plt.show()
visualize_clusters_center(cluster_centers_3)
visualize_clusters_center(cluster_centers_4)
visualize_clusters_center(cluster_centers_5)
# Saving data with labels for 6 clusters as the most promising

Data["cluster_3"] = cluster_labels_3["cluster"]

Data["cluster_4"] = cluster_labels_4["cluster"]

Data["cluster_5"] = cluster_labels_5["cluster"]



Data.to_csv("ESS_final.csv", index = False)