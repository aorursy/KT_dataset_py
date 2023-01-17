import numpy as np

from tqdm import tqdm

from scipy.spatial.distance import cosine

from sklearn.decomposition import PCA

%matplotlib notebook

import matplotlib.pyplot as plt

from tqdm import tqdm
f = open('../input/glove.6B.50d.txt')
embedding_values = {}

for line in tqdm(f):

    value = line.split(' ')

    word = value[0]

    coef = np.array(value[1:], dtype='float32')

    embedding_values[word] = coef
ix_to_word = {}

word_to_ix = {}



for word in tqdm(embedding_values):

    ix_to_word[len(ix_to_word)] = word

    word_to_ix[word] = len(ix_to_word)
def most_similar(word, count):

    cos = []

    for i in tqdm(embedding_values):

        cos.append(cosine(embedding_values[word], embedding_values[i]))

    temp = cos.copy()

    temp.sort()

    for i in range(count):

        id = cos.index(temp[i])

        print(ix_to_word[id])
most_similar('king', 10)
def analogy(word1, word2, word3):

    embeds = embedding_values[word2]+embedding_values[word3]-embedding_values[word1]



    cos = []

    for i in tqdm(embedding_values):

        cos.append(cosine(embeds, embedding_values[i]))



    idx = np.array(cos).argsort()[1]

    word4 = ix_to_word[idx]

    

    return word4

analogy('man', 'king', 'woman')
analogy('india', 'delhi', 'italy')