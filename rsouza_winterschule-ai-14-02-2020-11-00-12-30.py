! cp ../input/german-word2vec/myutils.py .

! ls -l
# Importing packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (20.0, 15.0)

import mpld3



import gensim

from gensim import corpora

from gensim.corpora import WikiCorpus

from gensim.models import Word2Vec

from gensim.models.word2vec import LineSentence

from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec

from gensim.test.utils import datapath, get_tmpfile



import networkx as nx



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from ipywidgets import IntProgress

from IPython.display import display



import myutils
# get trained model, files without a suffix, .bin or .model are treated as binary files

trained_model1 = gensim.models.KeyedVectors.load_word2vec_format('../input/german-word2vec/german.model', binary=True)

# remove original vectors to free up memory

trained_model1.init_sims(replace=True)
word = "Fruehstueck"

[k for k,w in trained_model1.vocab.items() if k.startswith(word)][0:20]
word = "Wien"

trained_model1[word]
print(trained_model1.similarity('kopf', 'blau'))

print(trained_model1.distance('kopf', 'blau'))
words = ['Werkzeug', 'blau', 'rot', 'kopf', 'Gewerbe']

series = []

for word in words:

    series.append(pd.DataFrame(trained_model1.most_similar(word, topn=10), columns=[f"Similar_{word}", "similarity"]))

df = pd.concat(series, axis=1)

df.head(10)
#word1, word2, word3, word4, word5 = 'blau','rot','feld','gruen','gelb'

#word1, word2, word3, word4, word5 = 'Fruehstueck', "Fenster", 'Abendessen','Mittagessen', "Soupe"

word1, word2, word3, word4, word5 = "Vater", "Mutter", "Sohn", "Tochter", "Oma"

#word1, word2, word3, word4, word5 = "Frankreich","England","Deutschland","Berlin","Oesterreich"







print(trained_model1.doesnt_match([word1, word2, word3, word4, word5]))
#positive_vectors = ['Koenig', 'frau']

#negative_vectors = ['mann']



positive_vectors = ['frau', 'blau']

negative_vectors = ['mann']



for result in trained_model1.most_similar(positive=positive_vectors, 

                                          negative=negative_vectors):

    print(result)
wordpairs = ["Mann", "Vater",

             "Frau",  "Mutter",

             "Mutter", "Oma",

             "Vater", "Grossvater",

             "Junge", "Mann",

             "Maedchen", "Frau",

            ]



myutils.draw_words(trained_model1, wordpairs, True, True, True, -2.5, 2.5, -2.5, 2.5, r'$PCA Visualisierung:')
# plot currencies

wordpairs = ["Schweiz", "Franken",

             "Deutschland", "Euro",

             "Grossbritannien", "britische_Pfund",

             "Japan", "Yen",

             "Russland", "Rubel",

             "USA", "US-Dollar",

             "Kroatien", "Kuna",

             "Oesterreich", "Euro",]



myutils.draw_words(trained_model1, wordpairs, True, True, True, -2, 2, -2, 2, r'$PCA Visualisierung:')
wordpairs = ["duerfen", "koennen",  # change this words and run the cell again

             "moegen", "muessen",

            ]



myutils.draw_words(trained_model1, wordpairs, True, True, True, -2.5, 2.5, -2.5, 2.5, r'$PCA Visualisierung:')
#word = 'Oesterreich'

#word = 'Akademie'

#word = "Kant"

word = "Rock"



G = myutils.build_neighbors(word, trained_model1, 10) # number of similar words to display

pos = nx.spring_layout(G, iterations=100)

nx.draw_networkx(G,

                 pos=pos, 

                 node_color=nx.get_node_attributes(G,'color').values(),

                 node_size=1000, 

                 alpha=0.8, 

                 font_size=12,

                )
! ls ../input/cookbooks/