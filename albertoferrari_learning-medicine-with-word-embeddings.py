import re

import pandas as pd

from collections import defaultdict

import spacy

import en_core_web_sm

import numpy as np

import matplotlib.pyplot as plt
metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

metadata=metadata[metadata["abstract"]==metadata["abstract"]]

print(metadata.shape)
nlp = en_core_web_sm.load(disable=['ner', 'parser']) # disabling Named Entity Recognition for speed



def cleaning(doc):

    # Lemmatizes and removes stopwords

    # doc needs to be a spacy Doc object

    txt = [token.lemma_ for token in doc if not token.is_stop]

    # Word2Vec uses context words to learn the vector representation of a target word,

    # if a sentence is only one or two words long,

    # the benefit for the training is very small

    if len(txt) > 2:

        return ' '.join(txt)

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower().replace("abstract", "") for row in metadata['abstract'])

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

df_clean = pd.DataFrame({'clean': txt})

df_clean = df_clean.dropna().drop_duplicates()
from gensim.models.phrases import Phrases, Phraser

sent = [row.split() for row in df_clean['clean']]

phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

sentences = bigram[sent]
word_freq = defaultdict(int)

for sent in sentences:

    for i in sent:

        word_freq[i] += 1
MIN_COUNT=6

DICT=[key for key in word_freq.keys() if word_freq[key]>MIN_COUNT]
"covid" in DICT
import multiprocessing



from gensim.models import Word2Vec



cores = multiprocessing.cpu_count()



w2v_model = Word2Vec(min_count=MIN_COUNT,

                     window=2,

                     size=200,

                     sample=1e-4, 

                     alpha=0.01,

                     min_alpha=0.0001, 

                     negative=15,

                     workers=cores-1)



w2v_model.build_vocab(sentences, progress_per=10000)



w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



w2v_model.init_sims(replace=True)
w2v_model.wv.most_similar("treatment",topn=10)
w2v_model.wv.most_similar(positive=["china"],topn=10)
w2v_model.wv.most_similar(positive=["oseltamivir"],topn=10)
w2v_model.wv.most_similar(positive=["malaria"],topn=10)
w2v_model.wv.most_similar(positive=["covid"],topn=10) #ncp=novel coronavirus pneumonia
w2v_model.wv.most_similar(positive=["remdesivir"],topn=20) #recommended by WHO on Jan 24
diseases=["covid","coronavirus","sars","mers","covid_pneumonia","novel_coronavirus","ncov","cov","coronavirus_sars","sars_cov","influenza","seasonal_influenza","h_n","swine_flu","avian_flu"]
X = w2v_model[diseases].T

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

result = pca.fit_transform(X.T)

plt.figure(figsize=(15,5))

plt.scatter(result.T[0],result.T[1])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel("Component 1",fontsize=14)

plt.ylabel("Component 2",fontsize=14)

for i in range(0,len(diseases)):

    plt.annotate(diseases[i],xy=(result.T[0][i],result.T[1][i]), xycoords='data',

            xytext=(-15, 5), textcoords='offset points', fontsize=15)