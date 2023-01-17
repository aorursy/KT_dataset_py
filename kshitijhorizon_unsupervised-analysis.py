import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/fine-food-cleaned/eda_final.csv')

data_train = data.cleaned_text.sample(5000)

data_train.tail()
# Initializing the tf-idf vectorizer

tfidf = TfidfVectorizer(ngram_range=(1,2),min_df=10)

data_features = tfidf.fit_transform(data_train)
# Now we will scale it 

std_Scaler = StandardScaler(with_mean=False)

std_data = std_Scaler.fit_transform(data_features)
# Defining a function to find the optimal k

def find_optimal_k(std_data):

    loss = []

    k = list(range(2, 15))

    for noc in k:

        model = KMeans(n_clusters = noc)

        model.fit(std_data)

        loss.append(model.inertia_)

    plt.plot(k, loss, "-o")

    plt.title("Elbow method to choose k")

    plt.xlabel("K")

    plt.ylabel("Loss")

    plt.show()
find_optimal_k(std_data)
# Kmeans to 

k_Means = KMeans(n_clusters = 5)

k_Means.fit(std_data)

pred = k_Means.predict(std_data)
k_Means.cluster_centers_.shape
# Plot each cluster features in a cloud

def plot_cluster_cloud(features, coef):

    coef_df = pd.DataFrame(coef, columns = features)

#     print(len(coef_df))

    # Create a figure and set of 15 subplots because our k range is in between 

    fig, axes = plt.subplots(3, 3, figsize = (30, 20))

    fig.suptitle("Top 20 words for each cluster ", fontsize = 50)

    cent = range(len(coef_df))

    for ax, i in zip(axes.flat, cent):

        wordcloud = WordCloud(background_color = "white").generate_from_frequencies(coef_df.iloc[i,:].sort_values(ascending = False)[0:20])

        ax.imshow(wordcloud)

        ax.set_title("Cluster {}".format(i+1), fontsize = 30)

        ax.axis("off")

    plt.tight_layout()
features = tfidf.get_feature_names()

coef = k_Means.cluster_centers_
plot_cluster_cloud(features, coef)
data_features.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)

reduced_features = pca.fit_transform(std_data.toarray())



reduced_cluster_centers = pca.transform(k_Means.cluster_centers_)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=k_Means.predict(std_data))

plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

plt.rcParams["figure.figsize"] = (8,4)
from sklearn.metrics import silhouette_score

silhouette_score(std_data, labels=k_Means.predict(std_data))
data_pos = data[data.score_pos_neg == 'positive'].sample(2500)

data_pos.tail()



data_neg = data[data.score_pos_neg == 'negative'].sample(2500)

data_neg.tail()



# type(data_pos)

data_train_equalized = pd.concat([data_pos,data_neg])

data_train_equalized= data_train_equalized.cleaned_text

data_train_equalized.shape
# Initializing the tf-idf vectorizer

tfidf_2 = TfidfVectorizer(ngram_range=(1,2),min_df=10)

data_features_2 = tfidf_2.fit_transform(data_train_equalized)
# Now we will scale it 

std_Scaler_2 = StandardScaler(with_mean=False)

std_data_2 = std_Scaler_2.fit_transform(data_features_2)
# Defining a function to find the optimal k

def find_optimal_k(std_data):

    loss = []

    k = list(range(2, 15))

    for noc in k:

        model = KMeans(n_clusters = noc)

        model.fit(std_data)

        loss.append(model.inertia_)

    plt.plot(k, loss, "-o")

    plt.title("Elbow method to choose k")

    plt.xlabel("K")

    plt.ylabel("Loss")

    plt.show()
find_optimal_k(std_data_2)
# Kmeans to 

k_Means_2 = KMeans(n_clusters = 6)

k_Means_2.fit(std_data_2)

pred_2 = k_Means_2.predict(std_data_2)
k_Means_2.cluster_centers_.shape
# Plot each cluster features in a cloud

def plot_cluster_cloud(features, coef):

    coef_df = pd.DataFrame(coef, columns = features)

#     print(len(coef_df))

    # Create a figure and set of 15 subplots because our k range is in between 

    fig, axes = plt.subplots(2, 5, figsize = (30, 20))

    fig.suptitle("Top 20 words for each cluster ", fontsize = 50)

    cent = range(len(coef_df))

    for ax, i in zip(axes.flat, cent):

        wordcloud = WordCloud(background_color = "white").generate_from_frequencies(coef_df.iloc[i,:].sort_values(ascending = False)[0:20])

        ax.imshow(wordcloud)

        ax.set_title("Cluster {}".format(i+1), fontsize = 30)

        ax.axis("off")

    plt.tight_layout()
features_2 = tfidf_2.get_feature_names()

coef_2 = k_Means_2.cluster_centers_
plot_cluster_cloud(features_2, coef_2)
data_features_2.shape
from sklearn.decomposition import PCA
pca_2 = PCA(n_components=2, random_state=0)

reduced_features_2 = pca.fit_transform(std_data_2.toarray())



reduced_cluster_centers_2 = pca.transform(k_Means_2.cluster_centers_)
plt.scatter(reduced_features_2[:,0], reduced_features_2[:,1], c=k_Means_2.predict(std_data_2))

plt.scatter(reduced_cluster_centers_2[:, 0], reduced_cluster_centers_2[:,1], marker='x', s=150, c='b')

plt.rcParams["figure.figsize"] = (10,4)
from sklearn.metrics import silhouette_score

silhouette_score(std_data_2, labels=k_Means_2.predict(std_data_2))