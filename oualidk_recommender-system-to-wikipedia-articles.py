# Imported libraries for dataframes, arrays, and plots

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

from urllib.request import urlretrieve

from scipy import sparse



# Imported libraries for preprocessing and modeling

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import NMF

from sklearn.manifold import TSNE
# Import the Wikipedia's articles dataset

url_words = 'https://raw.githubusercontent.com/jadoonengr/DataCamp-Notes/master/datasets/Wikipedia%20articles/wikipedia-vocabulary-utf8.txt'

words = np.genfromtxt(url_words, dtype='str')



url_articles = 'https://raw.githubusercontent.com/jadoonengr/DataCamp-Notes/master/datasets/Wikipedia%20articles/wikipedia-vectors.csv'

urlretrieve(url_articles, 'articles.csv')

articles = pd.read_csv('articles.csv', sep=',')

articles = articles.drop(articles.columns[0], axis=1)
# manipulate and transforme the articles dataset

articles = articles.transpose()

articles.columns = list(words)

art_csr = sparse.csr_matrix(articles)
# SVD dimentional reduction

def svd_reduction(data_csr, c, svd_index, iteration):

    svd = TruncatedSVD(n_components=c, n_iter=iteration)

    features = pd.DataFrame(svd.fit_transform(data_csr), index = svd_index)

    compo_var = svd.explained_variance_

    compo_var_ratio = svd.explained_variance_ratio_

    print('the '+str(c)+' constructed components explain over: {}'.format(round(compo_var_ratio.sum()*100, 2))+'%')

    return features

svd_features = svd_reduction(data_csr=art_csr, c=50, svd_index = list(articles.index), iteration=20)
# Elbow graph for optimal number of clusters

def elbow_graph(data, rang_nclusters):

    inertias = []

    for k in rang_nclusters:

        model = KMeans(n_clusters=k)

        model.fit(data)

        inertias.append(model.inertia_)

    plt.plot(rang_nclusters, inertias, 'o--')

    plt.xlabel('Number of clusters used')

    plt.ylabel('Amount of inertia')

    plt.title('The Elbow graph for optimal number of clusters')

    plt.show()

elbow_graph(data=svd_features, rang_nclusters=range(5,20))
# Fit a Kmeans clustering on svd_features

def mult_KMeans(svd_data, label_obs, rang_nclusters):

    articles_clus = pd.DataFrame({'articles':label_obs})

    for k in rang_nclusters:

        scaler = StandardScaler()

        kmeans=KMeans(n_clusters=k)

        pipeline=make_pipeline(scaler, kmeans)

        pred_clus = pipeline.fit_predict(svd_data)

        articles_clus.insert(k-rang_nclusters[0]+1, 'clus_k='+str(k), pred_clus, True)

        # Plot histogram for each clustering

        plt.figure(figsize=(20,20))

        sns.countplot(x='clus_k='+str(k), data=articles_clus)

        plt.xlabel('clusters', fontsize=18)

        plt.ylabel('number of articles in each cluster', fontsize=18)

        plt.title('KMEANS Clustering with k='+str(k), fontsize=18)

        plt.show()

    articles_clus.index=list(articles_clus.iloc[:,0])

    articles_clus = articles_clus.drop(articles_clus.columns[0], axis=1)

    return articles_clus

articles_clus = mult_KMeans(svd_data=svd_features, label_obs=list(articles.index), rang_nclusters=range(5,20))
# Fit a TSNE dimentional reduction to get component for better visualization

tsne_model = TSNE(n_components=2, perplexity=10, learning_rate=100)

tsne_features = tsne_model.fit_transform(svd_features)

tsne_features = pd.DataFrame(tsne_features, index = list(articles.index), columns=['cp0', 'cp1'])
# Create df dataframe that contain the components and the differents results of KMeans clustering

for col in articles_clus.columns:

    tsne_features[col]=articles_clus.loc[:,col]

df = tsne_features

xs = df.iloc[:,0]

ys = df.iloc[:,1]



# Visualize the clusters in a scatter plot using TSNE components:

for clus in list(df.columns[2:]):

    sns.lmplot(x='cp0', y='cp1', data=df, fit_reg=False, hue=clus, legend=False)

    for x, y, comp in zip(xs, ys, list(df.index)):

        plt.annotate(comp, (x,y), fontsize=7, alpha=0.8)

    plt.legend(loc='upper right')

    plt.show()
# Function of the recommander System

def syst_recommendation(data, c, k, ind, article): # data must be a csr format and ind is a list of index of the pandas data frame

    model = NMF(n_components=c)

    model.fit(data)

    nmf_features = model.transform(data)

    features = pd.DataFrame(nmf_features, index = ind) # ind here is the list of article's name

    norm_features = normalize(features)

    df = pd.DataFrame(norm_features, index=ind)

    current_article = df.loc[article]

    similiraty = df.dot(current_article)

    print(similiraty.nlargest(k))
# Fit a recommandation example of article 'Social search'

art_csr = sparse.csr_matrix(articles)    

syst_recommendation(data = art_csr, c=10, k=6, ind=list(articles.index), article = 'Social search')

syst_recommendation(data = art_csr, c=10, k=6, ind=list(articles.index), article = 'Cristiano Ronaldo')