# import dependencies

%matplotlib inline

import pandas as pd

import numpy as np

import multiprocessing

import nltk

from nltk.corpus import stopwords

from nltk import FreqDist

from collections import Counter

import random

import re

import matplotlib.pyplot as plt

import seaborn as sns

import gensim.models.word2vec as w2v

import sklearn.manifold

import time

sns.set_style("darkgrid")
df = pd.read_csv('fake.csv', usecols = ['uuid','author','title','text','language','site_url','country'])

df = df[df.language == 'english']

df['title'].fillna(value="", inplace=True)

df.dropna(axis=0, inplace=True, subset=['text'])

df = df.sample(frac=1.0) # shuffle the data

df.reset_index(drop=True,inplace=True)

df.head()
example_text = "Hi there! Good morning Mr. Smith. You should check out www.example.com, its a great website"

nltk.sent_tokenize(example_text)
def sent_tokenizer(text):

    """

    Function to tokenize sentences

    """

    text = nltk.sent_tokenize(text)

    return text



def sentence_cleaner(text):

    """

    Function to lower case remove all websites, emails and non alphabetical characters

    """

    new_text = []

    for sentence in text:

        sentence = sentence.lower()

        sentence = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", sentence)

        sentence = re.sub("[^a-z ]", "", sentence)

        sentence = nltk.word_tokenize(sentence)

        sentence = [word for word in sentence if len(word)>1] # exclude 1 letter words

        new_text.append(sentence)

    return new_text



def apply_all(text):

    return sentence_cleaner(sent_tokenizer(text))
t1 = time.time()

df['sent_tokenized_text'] = df['text'].apply(apply_all)

t2 = time.time()

print("Time to clean and tokenize", len(df), "articles:", (t2-t1)/60, "min")
df.head()
# create a list of all words using list comprehension

all_words = [word for item in list(df['sent_tokenized_text']) for word in item]

all_words = [subitem for item in all_words for subitem in item]
fdist = FreqDist(all_words)

len(fdist) # number of unique words
# show the most common words and their counts corpus wide

fdist.most_common(20)
# choose k and visually inspect the bottom 10 words of the top k

k = 50000

top_k_words = fdist.most_common(k)

top_k_words[-10:]
# choose k and visually inspect the bottom 10 words of the top k

k = 30000

top_k_words = fdist.most_common(k)

top_k_words[-10:]
# get document lengths to plot histogram

def doc_length(text):

    return len([word for sent in text for word in sent])
# document length

df['doc_len'] = df['sent_tokenized_text'].apply(doc_length)

doc_lengths = list(df['doc_len'])

df.drop(labels='doc_len', axis=1, inplace=True)
print("length of list:",len(doc_lengths),

      "\naverage document length", np.average(doc_lengths),

      "\nmaximum document length", max(doc_lengths))
# plot a histogram of document length

num_bins = 1000

fig, ax = plt.subplots(figsize=(12,6));

# the histogram of the data

n, bins, patches = ax.hist(doc_lengths, num_bins, normed=1)

ax.set_xlabel('Document Length (tokens)', fontsize=15)

ax.set_ylabel('Normed Frequency', fontsize=15)

ax.grid()

ax.set_xticks(np.logspace(start=np.log10(250),stop=np.log10(4000),num=7, base=10.0))

plt.xlim(0,4000)

ax.plot([np.average(doc_lengths) for i in np.linspace(0.0,0.0022,100)], np.linspace(0.0,0.0022,100), '-',

        label='average doc length')

ax.legend()

ax.grid()

fig.tight_layout()

plt.show()
all_sentences = list(df['sent_tokenized_text'])

all_sentences = [subitem for item in all_sentences for subitem in item]

all_sentences[:2] # print first 5 sentences
token_count = sum([len(sentence) for sentence in all_sentences])

print("The corpus contains {0:,} tokens".format(token_count)) # total words in corpus
num_features = 300 # number of dimensions

# if any words appear less than min_word_count amount of times, disregard it

# recall we saw that the bottom 10 of the top 30,000 words appear only 7 times in the corpus, so lets choose 10 here

min_word_count = 10

num_workers = multiprocessing.cpu_count()

context_size = 7 # window size around target word to analyse

downsampling = 1e-3 # downsample frequent words

seed = 1 # seed for RNG
# setting up model with parameters above

fake2vec = w2v.Word2Vec(

    sg=1,

    seed=seed,

    workers=num_workers,

    size=num_features,

    min_count=min_word_count,

    window=context_size,

    sample=downsampling

)
fake2vec.build_vocab(all_sentences)
print("Word2Vec vocabulary length:", len(fake2vec.wv.vocab))
# number of sentences

fake2vec.corpus_count
# train word2vec - this may take a minute...

fake2vec.train(all_sentences, total_examples=fake2vec.corpus_count, epochs=fake2vec.iter)
# dense 2D matrix of word vectors

all_word_vectors_matrix = fake2vec.wv.syn0
all_word_vectors_matrix.shape # .shape[0] are the top words we are considering in training word2vec
# train tsne model for visualisation

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

t1 = time.time()

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

t2 = time.time()

print("time to train TSNE on", all_word_vectors_matrix.shape[0], "word vectors:", (t2-t1)/60, "min")
# create a dataframe *points* to store the 2D embeddings of all words

points = pd.DataFrame(

    [

        (word, coords[0], coords[1])

        for word, coords in [

            (word, all_word_vectors_matrix_2d[fake2vec.wv.vocab[word].index])

            for word in fake2vec.wv.vocab

        ]

    ],

    columns=["word", "x", "y"]

)
def plot_region(x_bounds, y_bounds):

    """

    This function defines regions of the tsne map

    in which to zoom in on

    """

    slice = points[

        (x_bounds[0] <= points.x) &

        (points.x <= x_bounds[1]) & 

        (y_bounds[0] <= points.y) &

        (points.y <= y_bounds[1])

    ]

    

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))

    for i, point in slice.iterrows():

        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
points.head(10)
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12), title="TSNE Map of word2vec embeddings")
plot_region((3,5), (-53,-50))
plot_region((-36,-31), (-32,-27))
plot_region((44,47), (3,6))
plot_region((8,11), (38,42))
# similar word relations

def nearest_similarity_cosmul(start1, end1, end2):

    similarities = fake2vec.most_similar_cosmul(

        positive=[end2, start1],

        negative=[end1]

    )

    start2 = similarities[0][0]

    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))

    return start2
fake2vec.most_similar('muslim')
fake2vec.most_similar('trump')
fake2vec.most_similar('clinton')
nearest_similarity_cosmul("trump", "presidentelect", "clinton") # makes sense
nearest_similarity_cosmul("cancer", "body", "trump") # what?