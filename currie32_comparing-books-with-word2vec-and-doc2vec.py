import codecs

import glob

import logging

import multiprocessing

import os

import pprint

import re

import nltk

import gensim.models.word2vec as w2v

import sklearn.manifold

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import gensim

%pylab inline
# Create a list of all of our book files.

book_filenames = sorted(glob.glob("../input/*.rtf"))

print("Found books:")

book_filenames
# Read and add the text of each book to corpus_raw.

corpus_raw = u""

for book_filename in book_filenames:

    print("Reading '{0}'...".format(book_filename))

    with codecs.open(book_filename, "r", "utf-8") as book_file:

        corpus_raw += book_file.read()

    print("Corpus is now {0} characters long".format(len(corpus_raw)))

    print()
# Tokenize each sentence

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)
def sentence_to_wordlist(raw):

    '''Remove all characters except letters'''

    clean = re.sub("[^a-zA-Z]"," ", raw)

    words = clean.split()

    return words
# Clean the raw_sentences and add them to sentences.

sentences = []

for raw_sentence in raw_sentences:

    if len(raw_sentence) > 0:

        sentences.append(sentence_to_wordlist(raw_sentence))
# Take a look at a sentence before and after it is cleaned.

print(raw_sentences[5])

print(sentence_to_wordlist(raw_sentences[5]))
# Find the total number of tokens in sentences

token_count = sum([len(sentence) for sentence in sentences])

print("The book corpus contains {0:,} tokens".format(token_count))
# Set the parameteres for Word2Vec

num_features = 300

min_word_count = 20

num_workers = multiprocessing.cpu_count()

context_size = 10

downsampling = 1e-4

seed = 2
books2vec = w2v.Word2Vec(

    sg=1, #skip-gram

    seed=seed,

    workers=num_workers,

    size=num_features,

    min_count=min_word_count,

    window=context_size,

    sample=downsampling

)
# Build the vocabulary

books2vec.build_vocab(sentences)

print("books2vec vocabulary length:", len(books2vec.wv.vocab))
books2vec.train(sentences)
# Create a vector matrix of all the words

all_word_vectors_matrix = books2vec.wv.syn0
# Use TSNE to reduce all_word_vectors_matrix to 2 dimensions. 

tsne = sklearn.manifold.TSNE(n_components = 2, 

                             early_exaggeration = 6,

                             learning_rate = 500,

                             n_iter = 2000,

                             random_state = 2)
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
# Create a dataframe to record each word and its coordinates.

points = pd.DataFrame(

    [(word, coords[0], coords[1])

        for word, coords in [

            (word, all_word_vectors_matrix_2d[books2vec.wv.vocab[word].index])

            for word in books2vec.wv.vocab

        ]],

    columns=["word", "x", "y"])
# Preview the points

points[100:105]
# Display the layout of all of the points.

sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(10, 6))
def plot_region(x_bounds, y_bounds):

    '''Plot a limited region with points annotated by the word they represent.'''

    slice = points[(x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) & 

                   (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])]

    

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 6))

    for i, point in slice.iterrows():

        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
# Find the coordinates for Alice - Alice's Adventures in Wonderland

points[points.word == 'Alice']
plot_region(x_bounds=(0.5, 1.5), y_bounds=(3.2, 4.2))
# Find the coordinates for (Tom) Sawyer - The Adventures of Tom Sawyer

points[points.word == 'Sawyer']
plot_region(x_bounds=(-4.5, -3.5), y_bounds=(1.0, 2.0))
books2vec.most_similar("monster") 
books2vec.most_similar("Sherlock")
books2vec.most_similar("dog")
books2vec.most_similar("frightened")
def nearest_similarity_cosmul(start1, end1, start2):

    '''Find the word that completes the relationship.'''

    similarities = books2vec.most_similar_cosmul(

        positive=[start1, start2],

        negative=[end1])

    end2 = similarities[0][0]

    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))

    return end2
nearest_similarity_cosmul("Sherlock", "Holmes", "Aunt")

nearest_similarity_cosmul("Pip", "Havisham", "Alice")
# Read and tag each book into book_corpus

book_corpus = []

for book_filename in book_filenames:

    with codecs.open(book_filename, "r", "utf-8") as book_file:

        book_corpus.append(

            gensim.models.doc2vec.TaggedDocument(

                gensim.utils.simple_preprocess( # Clean the text with simple_preprocess

                    book_file.read()),

                    ["{}".format(book_filename)])) # Tag each book with its filename
# We can expand the vocabulary by setting the min_count to 3.

# Larger values for iter should improve the model's accuracy.

model = gensim.models.Doc2Vec(size = 300, 

                              min_count = 3, 

                              iter = 100)
model.build_vocab(book_corpus)

print("model's vocabulary length:", len(model.wv.vocab))
model.train(book_corpus)
model.docvecs.most_similar(12) #The_Adventures_of_Tom_Sawyer_by_Mark_Twain
model.docvecs.most_similar(11) # The_Adventures_of_Sherlock_Holmes_by_Arthur_Conan_Doyle.rtf
model.docvecs.most_similar(16) # The_Prince_by_Nicolo_Machiavelli.rtf
# Find the most similar book for each book

for book in book_filenames:

    most_similar = model.docvecs.most_similar(book)[0][0]

    print("{} - {}".format(book, most_similar))

    print()