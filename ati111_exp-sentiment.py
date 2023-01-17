# importation des librairies utiles #

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, LabeledSentence


import os
import re
import string

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import keras.backend as K

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer


# lecture du dataset qui servira d'entraînement et de test #

dataset=pd.read_csv('/kaggle/input/dataset/train.csv')
# Traitement des données #

def processing(sentence):
    lower_sentence=sentence.lower()   # enlever les majuscules #
    removed_numb=re.sub(r'\d+', '',lower_sentence)  # enlever les nombres #
    cleaned=removed_numb.translate(str.maketrans("", "", string.punctuation))  # enlever les ponctuations #
    return cleaned
# appliquer le traitement à l'intégralité du dataset #

for k in range(len(dataset)):
    dataset.at[k,'comment_text']=processing(dataset['comment_text'][k])


# subdivision des données entraînement et test #

target_labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

X_train=dataset.loc[:127000,'comment_text'].values
Y_train=dataset.loc[:127000,target_labels].values

X_test=dataset.loc[127001:,'comment_text'].values
Y_test=dataset.loc[127001:,target_labels].values


# Définition des paramètres et conversion des chaînes de caractères en entiers #

MAX_VOCAB_SIZE=20000    # taille du vocabulaire à considérer #
MAX_SEQUENCE_LENGTH=100  # longueur max de la séquence #

EMBEDDING_DIM=50  


sentence_tokenizer=Tokenizer(num_words=MAX_VOCAB_SIZE)

sentence_tokenizer.fit_on_texts(X_train)
sentence_tokenizer.fit_on_texts(X_test)

sentences_train=sentence_tokenizer.texts_to_sequences(X_train)
sentences_test=sentence_tokenizer.texts_to_sequences(X_test)

data_train=pad_sequences(sentences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test=pad_sequences(sentences_test, maxlen=MAX_SEQUENCE_LENGTH)

# Construction du modèle #

model=Sequential()
model.add(Embedding(MAX_VOCAB_SIZE, EMBDEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.1), metrics=['accuracy'])
# Entraînement du modèle #

m=model.fit(data_train, Y_train, batch_size=128, epochs=25, validation_split=0.2, verbose=2)
# Evaluation sur le dataset test #

score, acc = model.evaluate(data_test, Y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
# Plot the training process # 

plt.plot(m.history['loss'], label=['loss'])
plt.plot(m.history['val_loss'], label=['val_loss'])
plt.title('Loss evolution at each epochs')
plt.legend()
plt.show()

plt.plot(m.history['accuracy'], label=['accuracy'])
plt.plot(m.history['val_accuracy'], label=['val_accuracy'])
plt.title('Accuracy evolution at each epochs')
plt.legend()
plt.show()
# Second modèle, utilisant un word embedding pré-entraîné #

embedding_index={}

file=open(os.path.join('/kaggle/input/glove-data/glove.6B.50d.txt'))  # chargement du embedding pré-entraîné 
for line in file:
    values=line.split()
    word=values[0]
    coeff=np.asarray(values[1:], dtype='float32')
    embedding_index[word]=coeff
    
num_words=sentence_tokenizer.word_index     

embedding_matrix=np.zeros((len(num_words)+1,EMBEDDING_DIM))   # construction de embedding matrix #
for word, i in num_words.items():
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
        
# Définition du deuxième modèle #

model_2=Sequential()

# définition du embedding layer qui va être utilisé #

embedding_layer=Embedding(len(num_words)+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

model_2.add(embedding_layer)
model_2.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model_2.add(Dense(6, activation='softmax'))

model_2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.1), metrics=['accuracy'])
# Entraînement du second modèle utilisant Glove pré-entraîné #

m2=model_2.fit(data_train, Y_train, batch_size=128, epochs=25, validation_split=0.2, verbose=2)
# Evaluation sur le dataset test #

score, acc = model_2.evaluate(data_test, Y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
# Plot the training process # 

plt.plot(m2.history['loss'], label=['loss'])
plt.plot(m2.history['val_loss'], label=['val_loss'])
plt.title('Loss evolution at each epochs')
plt.legend()
plt.show()

plt.plot(m2.history['accuracy'], label=['accuracy'])
plt.plot(m2.history['val_accuracy'], label=['val_accuracy'])
plt.title('Accuracy evolution at each epochs')
plt.legend()
plt.show()
# calcul des n mots les plus utilisés, avec k-windows  #

def get_top_n_words(corpus,k,n=None):
    vec = CountVectorizer(ngram_range=(k, k),stop_words = 'english').fit(corpus)   # on enlève les stopwords #
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
# Préparation des mots pour l'entrainement du Word2Vec #

comments=dataset['comment_text'].values.tolist()
filtered_sentence=list()
stop_words=set(stopwords.words('english'))

for line in comments:
    token=word_tokenize(line)
    token=[w.lower() for w in token]
    table=str.maketrans('','',string.punctuation)
    stripped=[w.translate(table) for w in token]
    words=[word for word in stripped if word.isalpha()]
    words=[w for w in words if not w in stop_words]
    filtered_sentence.append(words)
  
filtered_sentence
# train Word2Vec embedding #

word2vec=Word2Vec(filtered_sentence, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)

# enregistrement du modèle #

store_file='pretrained_word2vec.txt'
word2vec.wv.save_word2vec_format(store_file,binary=False)
# chargement du word2vec entraîné #

embedding_index={}

file=open(os.path.join('/kaggle/input/word2vec/pretrained_word2vec.txt'))  # chargement du embedding pré-entraîné 
for line in file:
    values=line.split()
    word=values[0]
    coeff=np.asarray(values[1:], dtype='float32')
    embedding_index[word]=coeff
    
num_words=sentence_tokenizer.word_index     

embedding_matrix=np.zeros((len(num_words)+1,EMBEDDING_DIM))   # construction de embedding matrix #
for word, i in num_words.items():
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
        
# Définition du troisième modèle #

model_3=Sequential()

# définition du embedding layer qui va être utilisé #

embedding_layer=Embedding(len(num_words)+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

model_3.add(embedding_layer)
model_3.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model_3.add(Dense(6, activation='softmax'))

model_3.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.1), metrics=['accuracy'])
# Entraînement du troisème modèle utilisant word2vec pré-entraîné #

m3=model_3.fit(data_train, Y_train, batch_size=128, epochs=25, validation_split=0.2, verbose=2)
# Plot the training process # 

plt.plot(m3.history['loss'], label=['loss'])
plt.plot(m3.history['val_loss'], label=['val_loss'])
plt.title('Loss evolution at each epochs')
plt.legend()
plt.show()

plt.plot(m3.history['accuracy'], label=['accuracy'])
plt.plot(m3.history['val_accuracy'], label=['val_accuracy'])
plt.title('Accuracy evolution at each epochs')
plt.legend()
plt.show()
# Evaluation sur le dataset test #

score, acc = model_3.evaluate(data_test, Y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
# Text processing to train Doc2Vec embedding #


documents = [TaggedDocument(filtered_sentence,[i]) for i in range(0,len(filtered_sentence))]
#doc2vec = Doc2Vec(documents, vector_size=EMBEDDING_DIM, window=2, min_count=1, workers=4)

documents
