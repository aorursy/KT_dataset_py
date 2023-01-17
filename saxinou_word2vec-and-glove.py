# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#=================regex===============
import re
#=================nltk===============
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopword_list = nltk.corpus.stopwords.words('english')
""" 
Create your own stopwords list
"""
stopword_list += ['a', 'about', 'above', 'across', 'after', 'afterwards', 'also']
stopword_list += ['again', 'against', 'all', 'almost', 'alone', 'along','us','said', 'may', 'even']
stopword_list += ['this', 'is', 'your', 'must', 'many','would', 'could', 'like','much']

print("StopWords List in English : \n", stopword_list)

special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    
    text = " ".join(text)
    #Remove Special Characters
    text=special_character_removal.sub('',text)
    
    #Replace Numbers
    text=replace_numbers.sub('n',text)
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        text = [w for w in text if not w in stopword_list]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        
    # Return a list of words
    text = text.split()
    return(text)

""" """ """ """ """ """ """ """ """
Imports needed and Logging 
""" """ """ """ """ """ """ """ """
import gensim 
import logging
from nltk.corpus import brown, movie_reviews, treebank # Different corpus provide by nltk 

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
 
""" """ """ """ """ """ """ """ """
Choose your dataset : 
you should have lots and lots of text data in the relevant domain. 
For example, if I want to make a sentiment analysis, using wikipedia as corpus may not be effective.
""" """ """ """ """ """ """ """ """
print(">>> Get your data ...")
sentences = brown.sents()
print("Our text data : \n", sentences)

""" """ """ """ """ """ """ """ """
Preprocess your text data (if you want), to avoid noise
-> Remove stopwords
-> Stemming ...
""" """ """ """ """ """ """ """ """
print(">>> Preprocessing data...")
comments = []
for text in sentences:
    comments.append(text_to_wordlist(text,stem_words=False))
print("Before preprocessing data : ", sentences[0:2])
print("After preprocessing data : ", comments[0:2])

"""
Before preprocessing data :  ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']
After preprocessing data :  ['fulton', 'county', 'grand', 'jury', 'friday', 'investigation', 'atlantas', 'recent', 'primary', 'election', 'produced', 'evidence', 'irregularities', 'took', 'place']
"""

""" """ """ """ """ """ """ """ """
Build and train your model 
""" """ """ """ """ """ """ """ """
print(">>> Build Word2Vec model ...")
# >>> Training the Word2Vec model - Straightforward. 
model = gensim.models.Word2Vec(comments,
                               size=150,
                               window=10,
                               min_count=1)

# After building the vocabulary, we just need to call train() function to start training the Word2Vec model
print(">>> Train the model ...")
model.train(comments, 
            total_examples=len(comments), 
            epochs=10)

""" """ """ """ """ """ """ """ """
Some result 
""" """ """ """ """ """ """ """ """
# >>> Access all the term in vocabulary
vocab = list(model.wv.vocab.keys())
print(type(vocab)) # list

# >>> Get vector for word - vectorial representation of a particular term
model['men']  

# >>> Which words is similar to... ?
w1 = "men"
model.wv.most_similar(positive = w1, topn=5)

# w2 = "france"
# model.wv.most_similar(positive = w2)
# KeyError: "word 'france' not in vocabulary"

# >>> What is the similarity between two words ?
model.wv.similarity(w1 = "women", w2 = "men")   
import numpy as np
import re
import nltk

from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline


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

from sklearn.manifold import TSNE

def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
display_closestwords_tsnescatterplot(model, 'men')

""" 
MODEL 1
print(">>> Build Word2Vec model ...")
# >>> Training the Word2Vec model - Straightforward. 
model_baseline = gensim.models.Word2Vec(comments)
model_baseline.train(comments,  total_examples=len(comments), epochs=10)

model_1a = gensim.models.Word2Vec(comments, size = 200)
model_1a.train(comments,  total_examples=len(comments), epochs=10)

model_1b = gensim.models.Word2Vec(comments, size = 150)
model_1b.train(comments,  total_examples=len(comments), epochs=10)

model_1c = gensim.models.Word2Vec(comments, size = 50)
model_1c.train(comments,  total_examples=len(comments), epochs=10)

model_2 = gensim.models.Word2Vec(comments, window = 15)
model_2.train(comments,  total_examples=len(comments), epochs=10)

model_baseline.wv.similarity(w1 = "women", w2 = "men")   
model_1a.wv.similarity(w1 = "women", w2 = "men")   
model_1b.wv.similarity(w1 = "women", w2 = "men")   
model_1c.wv.similarity(w1 = "women", w2 = "men")   
model_2.wv.similarity(w1 = "women", w2 = "men")  
""" 
from nltk.corpus import brown, movie_reviews, treebank # Different corpus provide by nltk 

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

print(">>> Get your data ...")
sentences_brown = brown.sents()
sentences_movies = movie_reviews.sents()
sentences_tree = treebank.sents()

print(">>> Vizualize your data ...")
i = 3
print("\nSentence ", i+1," Brown Corpus : ", sentences_brown[i])
print("\nSentence ", i+1," Movies Corpus : ", sentences_movies[i])
print("\nSentence ", i+1," Treebank Corpus : ", sentences_tree[i])

print("\n>>> Categories")
print("\nCategories for Brown corpus ", brown.categories())
print("\nCategories for Movies corpus ", movie_reviews.categories())


""" """ """ """ """ """ """ """ """
Preprocess your text data (if you want), to avoid noise
-> Remove stopwords
-> Stemming ...
""" """ """ """ """ """ """ """ """

print(">>> Preprocessing data...")
comments_brown = []
comments_movies = []
comments_tree = []
for text in sentences_brown:
    comments_brown.append(text_to_wordlist(text,stem_words=False))
for text in sentences_movies:
    comments_movies.append(text_to_wordlist(text,stem_words=False))
for text in sentences_tree:
    comments_tree.append(text_to_wordlist(text,stem_words=False))


""" """ """ """ """ """ """ """ """
Build and train your model 
""" """ """ """ """ """ """ """ """

print(">>> Build Word2Vec model ...")
# >>> Training the Word2Vec model - Straightforward. 
model_brown = gensim.models.Word2Vec(comments_brown,
                               size=150,
                               window=10,
                               min_count=1)
model_movies = gensim.models.Word2Vec(comments_movies,
                               size=150,
                               window=10,
                               min_count=1)
model_tree = gensim.models.Word2Vec(comments_tree,
                               size=150,
                               window=10,
                               min_count=1)

# After building the vocabulary, we just need to call train() function to start training the Word2Vec model
print(">>> Train the model ...")
model_brown.train(comments_brown, 
            total_examples=len(comments_brown), 
            epochs=10)
model_movies.train(comments_movies, 
            total_examples=len(comments_movies), 
            epochs=10)
model_tree.train(comments_tree, 
            total_examples=len(comments_tree), 
            epochs=10)

""" 
Some result 
"""
print("Count words in brown corpus ",len(model_brown.wv.vocab.keys()))
print("Count words in Movies corpus ",len(model_movies.wv.vocab.keys()))
print("Count words in Treebank corpus ",len(model_tree.wv.vocab.keys()))

print("\nCosinus similarity with Brown Corpus ", 
      model_brown.wv.similarity(w1 = "women", w2 = "men"))
print("Cosinus similarity with Movies Corpus ", 
      model_movies.wv.similarity(w1 = "women", w2 = "men"))
print("Cosinus similarity with Treebank Corpus ", 
      model_tree.wv.similarity(w1 = "women", w2 = "men"))
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
vocab = list(model.wv.vocab)
X = model[vocab]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df = pd.concat([pd.DataFrame(X_pca),
                pd.Series(vocab)],
               axis=1)
df.columns = ['x','y','word']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['x'],df['y'])
plt.show()

"""
# Common word

vocab_brown = set(model_brown.wv.vocab.keys())
vocab_tree = set(model_tree.wv.vocab.keys())
common = list(vocab_brown.intersection(vocab_tree))

# Count words occurence in a corpus 

lst_mostcommon = []
for words in common:
    if model_brown.wv.vocab[words].count > 800:
        print("Word Brown: ", words , " appears " , model_brown.wv.vocab[words].count)
        lst_mostcommon.append(words)
    if model_tree.wv.vocab[words].count > 800:
        print("Word Tree: ", words , " appears " , model_tree.wv.vocab[words].count)
        lst_mostcommon.append(words)        
"""


if 'election' in model_tree.wv.vocab.keys():
    print ("ok")
# Use a filter : 
for doc in labeled_corpus:
    words = filter(lambda x: x in model.vocab, doc.words)



from gensim import utils, corpora, matutils, models
import glove
 
# Restrict dictionary to the 30k most common words.
wiki = models.word2vec.LineSentence('/data/shootout/title_tokens.txt.gz')
id2word = corpora.Dictionary(wiki)
id2word.filter_extremes(keep_n=30000)
word2id = dict((word, id) for id, word in id2word.iteritems())
 
# Filter all wiki documents to contain only those 30k words.
filter_text = lambda text: [word for word in text if word in word2id]
filtered_wiki = lambda: (filter_text(text) for text in wiki)  # generator
 
# Get the word co-occurrence matrix -- needs lots of RAM!!
cooccur = glove.Corpus()
cooccur.fit(filtered_wiki(), window=10)
 
# and train GloVe model itself, using 10 epochs
model_glove = glove.Glove(no_components=600, learning_rate=0.05)
model_glove.fit(cooccur.matrix, epochs=10)

