import pandas as pd

import numpy as np

import os

import sys

import math

from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold

#from sklearn import cross_validation
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
print(os.listdir('../input'))
path = '../input/aclimdb/aclImdb/'
import nltk

#nltk.download('punkt')

#nltk.download('stopwords')
#Add all the data ...

negpath  = os.listdir(path + 'train/neg')

pospath = os.listdir(path + 'train/pos')

print(len(negpath))

#comment out above two lines to not cut the lists shorter

                  
#for j in range(1, 12500):

tflist_neg = []

for i in negpath:

    f = open(path+'train/neg/'+i,"r",encoding="utf-8")

    file = f.read();

    #data = [(word.replace(",", "")

     #           .replace(".", "")

      #          .replace("(", "")

       #         .replace(")", "")

       #         .replace("&", ""))

       #        for word in file.lower().split()]

    #text = word_tokenize(file)     

    tflist_neg.append(file)



    

    #print(data)

    #break

    #data = data[1:]

    
#for j in range(1, 12500):

tflist_pos = []

for i in pospath:

    f = open(path+'train/pos/'+i,"r",encoding="utf-8")

    file = f.read();

    #data = [(word.replace(",", "")

     #           .replace(".", "")

      #          .replace("(", "")

       #         .replace(")", "")

       #         .replace("&", ""))

       #        for word in file.lower().split()]

    #text = word_tokenize(file)     

    tflist_pos.append(file)



    

    #print(data)

    #break

    #data = data[1:]

    
trainDF = pd.DataFrame()

trainDF['text'] = tflist_neg

trainDF['label'] = 0
trainDF.head()
trainDF2 = pd.DataFrame()

trainDF2['text'] = tflist_pos

trainDF2['label'] = 1
trainDF = trainDF.append(trainDF2,ignore_index=True)
from sklearn.utils import shuffle

import random

trainDF_shuffled = shuffle(trainDF, random_state = 27)
trainDF_shuffled.head()
len(trainDF_shuffled)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF_shuffled['text'], trainDF_shuffled['label'], test_size=0.25, stratify = trainDF_shuffled['label'])
skf = StratifiedKFold(n_splits = 10, random_state = None)

trainDF_text = trainDF_shuffled['text']

trainDF_label = trainDF_shuffled['label']

for train, test in skf.split(trainDF_text, trainDF_label):

    X_train = trainDF_text[train]

    X_test = trainDF_text[test]

    Y_train = trainDF_label[train]

    Y_test = trainDF_label[test]

    
#prepare the text corpus for learning the embedding by creating word tokens, removing punctuation, removtk.coring stop words etc.

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



review_lines = list()

lines = trainDF_text

for line in lines:

    tokens = word_tokenize(line)

    tokens = [w.lower() for w in tokens]

    #remove punctuation from each word

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    #remove remaining tokens that are not alphabetic 

    words = [word for word in stripped if word.isalpha()]

    #filter out stop words

    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]

    review_lines.append(words)
len(review_lines)
import gensim

#train word2vec model

model = gensim.models.Word2Vec(sentences = review_lines, size=50, window=5, workers=4, min_count=1)

#vocab size

words = list(model.wv.vocab)

print('vocabulary size: %d' % len(words))
model.most_similar('good')
#save model

filename = 'TextDataProcessing_Word2Vec.txt'

model.wv.save_word2vec_format(filename, binary=False)
# extracting embeddings

import os

embeddings_index = {}

f = open(os.path.join('', 'TextDataProcessing_Word2Vec.txt'), encoding = 'utf-8')

for line in f:

    values  = line.split()

    word = values[0]

    coefs = np.asarray(values[1:])

    embeddings_index[word] = coefs

print(embeddings_index[word])

f.close()



    
import tensorflow

import keras

import keras.preprocessing.text as kpt
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(review_lines)

#total_reviews = X_train + X_test

#print(len(total_reviews))

sequences = tokenizer_obj.texts_to_sequences(review_lines) 

#pad sequences

word_index = tokenizer_obj.word_index

len(word_index)

review_pad = pad_sequences(sequences, maxlen= max(len(list) for list in review_lines))

sentiment = trainDF_shuffled['text'].values

print(review_pad.shape)

print(sentiment.shape)

#print(sequences[0])
#Now we will map embeddings from the loaded word2vec model for each word to the tokenizer_obj.word_index vocabulary and create a matrix with of word vectors.

num_words = len(word_index) + 1

EMBEDDING_DIM = 50

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():

  #print(word,i)

  if i > num_words:

    continue

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:

    #print(embedding_vector[0])

    #break

    #words not found in embeddings_index will be all zeros

    embedding_matrix[i] = embedding_vector

    



  
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU

from keras.layers.embeddings import Embedding

from keras.initializers import Constant



#define model

model = Sequential()

max_length= max(len(list) for list in review_lines)

embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=max_length, trainable=False)

model.add(embedding_layer)

model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



#try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
#train the sentiment classification model

#split the data into training set and validation set

VALIDATION_SPLIT = 0.2

#indices = np.arange(review_pad.shape[0])

#np.random.shuffle(indices)

#review_pad = review_pad[indices]

#sentiment = sentiment[indices]

num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

num_validation_labels = int(VALIDATION_SPLIT * trainDF_label.shape[0])



X_train_pad = review_pad[:-num_validation_samples]

y_train = trainDF_label[:-num_validation_labels]

X_test_pad = review_pad[-num_validation_samples:]

y_test = trainDF_label[-num_validation_labels:]

review_pad.shape[0]
model.fit(X_train_pad, y_train, batch_size=100, epochs=6, verbose=1) 

#validation_data=(X_test_pad, y_test), verbose=2)
model.save("/kaggle/working/model_neural.h5")