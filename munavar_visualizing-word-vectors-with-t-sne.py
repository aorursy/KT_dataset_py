import pandas as pd

pd.options.mode.chained_assignment = None 

import numpy as np

import re

import nltk



from gensim.models import word2vec



from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline



data = pd.read_csv('../input/train.csv').sample(50000, random_state=23)
STOP_WORDS = nltk.corpus.stopwords.words()



def clean_sentence(val):

    "remove chars that are not letters or numbers, downcase, then remove stop words"

    regex = re.compile('([^\s\w]|_)+')

    sentence = regex.sub('', val).lower()

    sentence = sentence.split(" ")

    

    for word in list(sentence):

        if word in STOP_WORDS:

            sentence.remove(word)  

            

    sentence = " ".join(sentence)

    return sentence



def clean_dataframe(data):

    "drop nans, then apply 'clean_sentence' function to question1 and 2"

    data = data.dropna(how="any")

    

    for col in ['question1', 'question2']:

        data[col] = data[col].apply(clean_sentence)

    

    return data



data = clean_dataframe(data)

data.head(5)
def build_corpus(data):

    "Creates a list of lists containing words from each sentence"

    corpus = []

    for col in ['question1', 'question2']:

        for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



corpus = build_corpus(data)        

corpus[0:2]
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model)
# A more selective model

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)

tsne_plot(model)
# A less selective model

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=100, workers=4)

tsne_plot(model)
model.most_similar('trump')
model.most_similar('universe')