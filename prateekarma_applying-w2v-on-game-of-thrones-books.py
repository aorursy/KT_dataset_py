from __future__ import absolute_import, division, print_function
import codecs 
import glob 
import multiprocessing
import os 
import pprint
import re 
import nltk 
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib as plt
import pandas as pd 
import seaborn as sns
nltk.download('punkt')#This is a pretrained tokenizer 
nltk.download('stopwords')
#To get the book
book_filenames = sorted(glob.glob('../input/game-of-thrones-books-for-nlp/*.txt'))
book_filenames
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'......".format(book_filename))
    with codecs.open(book_filename,"r","utf-8") as book_file:
        corpus_raw += book_file.read()
    print("corpus is now {0} characters long".format(len(corpus_raw)))
#     print()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
len(raw_sentences)
#to convert into a list of words :
#Plus removal of unnecessary characters like "[,/,\,#," etc. 
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ",raw)
    words = clean.split()
    return words
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence)>0:
        sentences.append(sentence_to_wordlist(raw_sentence))
token_count = sum([len(sentance) for sentance in sentences])
print("The book contains {0:,} tokens".format(token_count))
#DISTANCE, SIMILARITY, RANKING is what we get amongst words when we get the vectors out of text
dimensionality = 300
min_word_count = 3 #min num of word count threshold - "what is the smallest set of words that we want to choose to train"
cpu_cores = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3 #threshold for downweighing or reducing the importance of more frequent words 
seed = 1 #random number generator 
thrones2vec = w2v.Word2Vec(
    sg =1,
    seed = seed,
    workers = cpu_cores,
    size = dimensionality,
    min_count = min_word_count,
    window = context_size,
    sample = downsampling
    
)
thrones2vec.build_vocab(sentences)
# thrones2vec.train(sentences,total_words=thrones2vec.corpus_count,epochs=thrones2vec.epochs)
thrones2vec.train(sentences=sentences,total_examples=thrones2vec.corpus_count, epochs=10)
print("The shape of the matrix we are plugging into TSNE is : ",thrones2vec.wv.syn0.shape)
tsne = sklearn.manifold.TSNE(n_components=2,random_state=0)
word_vec_matrix = thrones2vec.wv.syn0
word_vec_matrix_in_2d = tsne.fit_transform(word_vec_matrix)
word_vec_matrix_in_2d[thrones2vec.wv.vocab['stark'].index]
points = pd.DataFrame(
    [
        (word,coordinates[0],coordinates[1])
        for word,coordinates in [
            (word,word_vec_matrix_in_2d[thrones2vec.wv.vocab[word].index]) 
            for word in thrones2vec.wv.vocab 
        ]
    ],columns = ["word", "x", "y"])
points.head(10)
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
plot_region(x_bounds=(0,3),y_bounds=(-3,3))
def linear_relationship_between_words(start1,end1,end2):
    similarities = thrones2vec.most_similar_cosmul(positive=[end2,start1],negative=[end1])
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2
linear_relationship_between_words("Cersei", "Lannister", "Jon")
linear_relationship_between_words("Arya","Jon", "Cersei")

thrones2vec.most_similar('Lannister')
thrones2vec.most_similar('Stark', topn=20)