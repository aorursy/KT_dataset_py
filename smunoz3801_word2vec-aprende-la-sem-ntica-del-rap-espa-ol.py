from __future__ import absolute_import, division, print_function



import codecs

import glob

import logging

import multiprocessing

import os

import pprint

import re

import string



import nltk

import gensim.models.word2vec as w2v

from sklearn.manifold import TSNE

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import gensim

from nltk.corpus import stopwords

#nltk.download("stopwords")
stopwords = set(stopwords.words('spanish'))
hhgroups = pd.read_csv('../input/hhgroups_merge_28_05.csv')

hhgroups.head()
letras = list(hhgroups['letra'])

letras_limpio = []

# Eliminamos canciones sin letra

for letra in list(letras):

    if "¿Tienes ya la letra para este tema? Ayúdanos y ¡Envíanosla!" in letra:

        letras.remove(letra)



#Eliminamos líneas vacías o con [Artista] o [Estribillo]

for i in range(len(letras)):

    cancion_limpia = []

    for linea in letras[i].split("\n"):

        if ("[" not in linea and "(" not in linea and linea != ""):

            #Pasamos línea a minúsculas y eliminamos puntuación

            linea = bytes(linea, 'utf-8').decode('utf-8', 'ignore')

            linea = "".join(c for c in linea if (c not in string.punctuation and c not in ['','¡','¿'])).lower()

            linea = linea.split(" ")

            #Eliminamos stopwords

            for palabra in list(linea):

                #palabra = palabra.replace(u'\xa0', u'') #Estp les pasa por usar latin en vez de UTF-8

                if palabra in stopwords or palabra in string.punctuation:

                    linea.remove(palabra)

            cancion_limpia += linea

    letras_limpio += [cancion_limpia]
letras_limpio[0][:10]
len(letras_limpio)
hhgroups2vec = w2v.Word2Vec(

    letras_limpio,

    sg=1,

    seed=1,

    workers=multiprocessing.cpu_count(),

    size=256,

    min_count=50,

    window=12

)
hhgroups2vec.wv.most_similar("familia")
hhgroups2vec.wv.most_similar("españa")
hhgroups2vec.wv.most_similar("1")
hhgroups2vec.wv.most_similar("arte")
hhgroups2vec.wv.most_similar("música")
hhgroups2vec.wv.most_similar("joder")
hhgroups2vec.wv.most_similar("bien")
def nearest_similarity_cosmul(start1, end1, end2):

    similarities = hhgroups2vec.wv.most_similar_cosmul(

        positive=[end2, start1],

        negative=[end1]

    )

    start2 = similarities[0][0]

    print("{0} es a {1}, lo que {2} es a {3}".format(start1, end1, start2, end2))
nearest_similarity_cosmul("ser", "siendo", "haciendo")
nearest_similarity_cosmul("amar", "amor", "odio")
nearest_similarity_cosmul("europa", "españa", "argentina")
hhgroups2vec.wv.doesnt_match("felicidad amor alegría envidia".split(" "))
hhgroups2vec.most_similar(positive=['mujer', 'rey'], negative=['hombre'])[2:]
def tsne_plot(model):

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(64, 64)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()

    

# call the function on our dataset

tsne_plot(hhgroups2vec)