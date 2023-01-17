"Importing necessary packages"

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

from gensim.models import LdaModel

from gensim.corpora import Dictionary

import pyLDAvis.gensim

pyLDAvis.enable_notebook()



#pd.set_option('display.expand_frame_repr', False)



import sklearn

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score

from sklearn import preprocessing

from mean_w2v import MeanEmbeddingVectorizer #custom function



import ipywidgets as widgets

from ipywidgets import Button, Layout

from IPython.display import display, HTML, clear_output



from lime import lime_text

from lime.lime_text import LimeTextExplainer



import warnings

warnings.filterwarnings("ignore")



"Prep dataset for LDA topic modelling"

dfeng = pd.read_csv("../input/rp-manglish-tweets-on-kl-rapid-transit//Eng_traindata_processed.csv")

dfmal = pd.read_csv("../input/rp-manglish-tweets-on-kl-rapid-transit//Malay_traindata_processed.csv")

dfeng['textfin'] = dfeng['textfin'].astype('str')

dfmal['textfin'] = dfmal['textfin'].astype('str')

corpus_mal = dfmal['textfin'].tolist()

corpus_eng = dfeng['textfin'].tolist()

train_eng_texts = [doc.split(" ") for doc in corpus_eng]

train_mal_texts = [doc.split(" ") for doc in corpus_mal]

dfmal_txt = pd.DataFrame(dfmal[['text','textfin']]);df_comb = pd.DataFrame(dfeng[['text','textfin']])

df_comb = df_comb.append(dfmal_txt, ignore_index = True)



"pyldaviz function. pyldaviz provides interactive visualization of topic modelling results "

def show_pyldavis(docs,passes,num_topics,no_below=0):  

    bigram = gensim.models.Phrases(docs, min_count=5, threshold=100) # higher threshold fewer phrases.

    bigram_mod = gensim.models.phrases.Phraser(bigram)

    texts = [bigram_mod[doc] for doc in docs]  

    dictionary = Dictionary(texts)

    dictionary.filter_extremes(no_below=no_below)

    dictionary.compactify()  

    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, 

                                        id2word=dictionary,

                                        random_state=100,

                                        chunksize=100,

                                        passes=passes,

                                        alpha="asymmetric",

                                        eta=0.91)

    viz = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

    return pyLDAvis.display(viz)
#print("Visualisation of LDA model of Manglish tweets")

#print("Instruction:")

#print("1) Hover your mouse on the circles to explore the topic-terms distribution")

#print("2) You may also explore the top terms across all topics by hover the mouse over to the right side panel")

#print("3) Event tweet is detected in the first topic/Topic 1. It is the biggest circle plot with the highest terms prevalence")

pd.options.display.max_colwidth = 50000

show_pyldavis(train_eng_texts,50,5,no_below=4)
#print("Visualisation of LDA model of English tweets")

#print("Instruction:")

#print("1) Hover your mouse on the circles to explore the topic-terms distribution")

#print("2) You may also explore the top terms across all topics by hover the mouse over to the right side panel")

#print("3) Event tweet is detected in the first topic/Topic 1. It is the biggest circle plot with the highest terms prevalence")

#pd.options.display.max_colwidth = 5000

show_pyldavis(train_mal_texts,50,8)