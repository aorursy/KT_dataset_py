# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

from gensim.models import Word2Vec

from sklearn.manifold import TSNE

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
f = open("../input/hurriyet.txt", "r", encoding = "utf8")

text = f.read()
t_list = text.split("\n")

corpus = []

for cumle in t_list:

    corpus.append(cumle.split())
print(corpus[:10])
model = Word2Vec(corpus, size = 100, window = 5, min_count= 5, sg = 1)
model.wv["ankara"]
model.wv.most_similar("hollanda")
model.wv.most_similar("cuma")
model.wv.most_similar("youtube")
model.save("word2vec.model")
model2 = Word2Vec.load("word2vec.model")
model2.wv.most_similar("cumartesi")
def closestwords_tsneplot(model, word):

    word_vectors = np.empty((0,100))

    word_labels = [word]

    close_words = model.wv.most_similar(word)

    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis = 0)

    

    for w, _ in close_words:

        word_labels.append(w)

        word_vectors = np.append(word_vectors, np.array([model.wv[w]]), axis = 0)

    tsne = TSNE(random_state = 0)

    Y = tsne.fit_transform(word_vectors)

    x_coords = Y[:,0]

    y_coords = Y[:,1]

    plt.scatter(x_coords, y_coords)

    

    for label, x, y in zip(word_labels, x_coords, y_coords):

        plt.annotate(label, xy = (x,y), xytext = (5, -2), textcoords = "offset points")

    

    plt.show()
closestwords_tsneplot(model, "berlin")