import os, string

from __future__ import print_function

from gensim.models import KeyedVectors

import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):

    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

    plt.figure(figsize=(18, 18))  # in inches

    for i, label in enumerate(labels):

        x, y = low_dim_embs[i, :]

        plt.scatter(x, y)

        plt.annotate(label,

                 xy=(x, y),

                 xytext=(5, 2),

                 textcoords='offset points',

                 ha='right',

                 va='bottom')

    plt.savefig(filename)
def clean_str(text):

    

    text = text.translate(string.punctuation)



    return text
# Limita o número de tokens que serão visualizados

limit = 200

# dimensionalidade do word vector

vector_dim = 50



#filename = "../input/word2vec-google/GoogleNews-vectors-negative{0}.bin".format(vector_dim)

filename = "../input/word2vec-cbow-50/cbow_s50.txt"



model = KeyedVectors.load_word2vec_format(filename, binary=False)





# Obtendo Tokens e vetores

words = []

embedding = np.array([])

i = 0

for word in model.vocab:

    # Interrompe o loop se o limite exceder

    if i == limit: break

        

    words.append(clean_str(word))

    embedding = np.append(embedding, model[word])

    i += 1

    

embedding = embedding.reshape(limit, vector_dim)
tsne = TSNE(n_components=2)

low_dim_embedding = tsne.fit_transform(embedding)



# plota o gráfico

plot_with_labels(low_dim_embedding, words)