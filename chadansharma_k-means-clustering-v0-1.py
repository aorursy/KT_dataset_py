# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Let's import the required packages 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm



from sklearn.cluster import MiniBatchKMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
#Import the data

data = pd.read_json('../input/department-of-justice-20092018-press-releases/combined.json', lines=True)

data.head()
#Convert the contents column into tfidf metrics

tfidf = TfidfVectorizer(

    min_df = 5,

    max_df = 0.95,

    max_features = 8000,

    stop_words = 'english'

)

tfidf.fit(data.contents)

text = tfidf.transform(data.contents)
#To convert the tfidf into df and see first few records

df = pd.DataFrame(text.toarray(),columns=[tfidf.get_feature_names()])

df[:5]
#Finding the optimal cluster through Elbow method

def find_optimal_clusters(data, max_k):

    iters = range(2, max_k+1, 2)

    

    sse = []

    for k in iters:

        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)

        print('Fit {} clusters'.format(k))

        

    f, ax = plt.subplots(1, 1)

    ax.plot(iters, sse, marker='o')

    ax.set_xlabel('Cluster Centers')

    ax.set_xticks(iters)

    ax.set_xticklabels(iters)

    ax.set_ylabel('SSE')

    ax.set_title('SSE by Cluster Center Plot')

    

find_optimal_clusters(text, 20)
#To see the sum of squared error for given cluster size

MiniBatchKMeans(n_clusters=16, init_size=1024, batch_size=2048, random_state=20).fit(text).inertia_
clusters = MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)
#Plot the clusters with PCA and tSNE graph

def plot_tsne_pca(data, labels):

    max_label = max(labels)

    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())

    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))

    

    

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)

    label_subset = labels[max_items]

    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]

    

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)

    ax[0].set_title('PCA Cluster Plot')

    

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)

    ax[1].set_title('TSNE Cluster Plot')

    

plot_tsne_pca(text, clusters)
def get_top_keywords(data, clusters, labels, n_terms):

    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    

    for i,r in df.iterrows():

        print('\nCluster {}'.format(i))

        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

            

get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)



#To join the clusters in original data

data['clusters'] = clusters
print(data.title[data.clusters==4])

# print(data.title[470])

# print(data.contents[470])