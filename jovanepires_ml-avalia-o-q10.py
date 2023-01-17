# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans



from warnings import filterwarnings

filterwarnings('ignore')



sns.set()

sns.set_palette("gist_rainbow")
# carregue os dados contidos no Dataset de Câncer do scikit-learn

X, y = load_breast_cancer(return_X_y=True)
# aplicar pré-processamento

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
# grafico com os dados sem clusters

plt.figure(figsize=(10,8))

plt.title("Raw data", fontsize=16)

sns.scatterplot(x=X[:, 0], y=X[:, 1], s=80, alpha=0.90, palette='tab10')

plt.show()
# gráfico com clusters

params = [2, 5, 10]



for i in params:

    kmeans = KMeans(n_clusters=i).fit(X)

    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    totals = dict(zip(unique, counts))

    

    plt.figure(figsize=(10,8))

    plt.title("KMeans: n_clusters={0}".format(i), fontsize=16)

    sns.set()

    ax = sns.scatterplot(x=X[:, 0],

                         y=X[:, 1],

                         hue=kmeans.labels_,

                         s=80,

                         alpha=0.90,

                         palette='tab10',

                         legend='full')

    

    handles, labels = ax.get_legend_handles_labels()

    labels = [int(l) for l in labels] # convert str labels to int

    labels_with_total = ["Cluster {0:02d} = {1}".format(l+1, totals[l]) for l in labels]

    plt.legend(title="Cluster x Total",

               handles=handles,

               labels=labels_with_total)



    plt.show()