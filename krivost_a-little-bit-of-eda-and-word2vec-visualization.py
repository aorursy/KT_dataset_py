import numpy as np 

import pandas as pd 
import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
import spacy

import re

nlp = spacy.load('en',disable=['parser', 'ner', 'textcat'])

def reduce_to_double_max(text):

    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)

    return re.sub(r'(\W)\1+', r'\1', text)

def preprocess_corpus(corpus):

    corpus = (reduce_to_double_max(s.lower()) for s in corpus)

    docs = nlp.pipe(corpus, batch_size=1000, n_threads=4)

    return [' '.join([x.lemma_ for x in doc if x.is_alpha]) for doc in docs]
train_processed = preprocess_corpus(train['comment_text'])

test_processed = preprocess_corpus(test['comment_text'])

train['comment_text'] = train_processed

test['comment_text'] = test_processed

train.head()
train.shape
train.dtypes
train.isnull().sum()
train['toxic_score'] = train['toxic'] + train['severe_toxic'] + train['obscene'] + train['threat'] + train['insult'] + train['identity_hate'] 
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(7, 5))

sns.countplot(x='toxic_score', data=train)
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
import re

import gensim

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import re 

from collections import namedtuple

import multiprocessing

import datetime

import os



tokenizer = RegexpTokenizer(r'\w+')

stopwords = stopwords.words("english")

lemmatizer = WordNetLemmatizer()

def get_processed_text(text=""):

    clean_text = re.sub('[^a-zA-Z \n\.]', ' ', text)

    tokens = tokenizer.tokenize(clean_text)

    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens

              if token not in stopwords and len(token) >= 2]

    return tokens
get_processed_text(train.comment_text[0])
tokens = []

for i in train.comment_text:

    tokens.append(get_processed_text(i))

tokens[0]
model_w2v = gensim.models.Word2Vec(tokens, window=5, min_count=5, size=50, seed=0)

model_w2v.train(sentences = tokens, total_examples=len(tokens), epochs=30)
model_w2v.wv.most_similar('thank')
def words_for_tsne(tokenized):

    count_vectorizer = CountVectorizer(max_features=1000)

    count_vec = count_vectorizer.fit_transform(tokenized)

    words = count_vectorizer.vocabulary_.keys()

    return words
from sklearn.feature_extraction.text import CountVectorizer

ws = words_for_tsne(train['comment_text'])
ws1 = []

for i in list(ws):

    if i not in stopwords and i not in ['discuss','pass', 'jews', 'less', 'ㅂㄱㅇ']:

        ws1.append(i)
from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, LabelSet, HoverTool

from bokeh.plotting import figure

from bokeh.io import show, output_notebook

output_notebook()

words_top_vec = model_w2v.wv[ws1]

tsne = TSNE(n_components=2, random_state=0)

words_top_tsne = tsne.fit_transform(words_top_vec)

p = figure(tools="pan,wheel_zoom,reset,save",

           toolbar_location="above",

           title="Word2Vec визуализация t-SNE для 1000 самых встречаемых слов")



source = ColumnDataSource(data=dict(x1=words_top_tsne[:,0],

                                    x2=words_top_tsne[:,1],

                                    names=ws1))



p.scatter(x="x1", y="x2", size=8, source=source)



labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,

                  text_font_size="8pt", text_color="#555555",

                  source=source, text_align='center')

p.add_layout(labels)

show(p)