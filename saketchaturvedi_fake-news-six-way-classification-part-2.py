import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import nltk

import string

import re

stopwords = nltk.corpus.stopwords.words('english')

import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

import gensim.models.keyedvectors as word2vec

import gc

from nltk.tokenize import sent_tokenize

from keras.optimizers import SGD, Adam

import itertools

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
# Check the contents of the input directory

os.listdir('../input/')
# Load the training, test and validation files

train=pd.read_csv('../input/liarplus/liar-plus-master/LIAR-PLUS-master/dataset/train2.tsv',delimiter='\t',encoding='utf-8', header=None)

test=pd.read_csv('../input/liarplus/liar-plus-master/LIAR-PLUS-master/dataset/test2.tsv',delimiter='\t',encoding='utf-8', header=None)

valid=pd.read_csv('../input/liarplus/liar-plus-master/LIAR-PLUS-master/dataset/val2.tsv',delimiter='\t',encoding='utf-8', header=None)



# Create table headers    

train.columns = ['values','id','label','statement','subject','speaker', 'job', 'state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue','extracted_justification']

test.columns = ['values','id','label','statement','subject','speaker', 'job', 'state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue','extracted_justification']

valid.columns = ['values','id','label','statement','subject','speaker', 'job', 'state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue','extracted_justification']



# reserve training and validation files for Training Set to provide enough examples for training

df = pd.concat([train, valid]) # training data

len(df)
# Copy the contents of two columns "statement" and "extracted_justification" to new column "text" of training data

df['text'] = df['statement'] + ' ' + df['extracted_justification'] 

df.head()
# Copy the contents of two columns "statement" and "extracted_justification" to new column "text" of test data

test['text'] = test['statement'] + ' ' + test['extracted_justification'] 

test1 = test

test.head() # test data
# print the sample contents of "text" column from training data

sample = df.text.values[0]

print(sample)
truth_ = {'false':0,'true':1,'half-true':2 ,'mostly-true':3 ,'barely-true':4 ,'pants-fire':5} # values for labels

df['num_true'] = df['label'].apply(lambda x: truth_[x])

df['num_false'] = df['label'].apply(lambda x: truth_[x])

df['num_half_true'] = df['label'].apply(lambda x: truth_[x])

df['num_mostly_true'] = df['label'].apply(lambda x: truth_[x])

df['num_barely_true'] = df['label'].apply(lambda x: truth_[x])

df['num_pants_fire'] = df['label'].apply(lambda x: truth_[x])

df.head() # Check the new columns just created
test['num_true'] = test['label'].apply(lambda x: truth_[x])

test['num_false'] = test['label'].apply(lambda x: truth_[x])

test['num_half_true'] = test['label'].apply(lambda x: truth_[x])

test['num_mostly_true'] = test['label'].apply(lambda x: truth_[x])

test['num_barely_true'] = test['label'].apply(lambda x: truth_[x])

test['num_pants_fire'] = test['label'].apply(lambda x: truth_[x])

test.head() # Check the new columns just created
# define the type of contents in the "text" column of training and test data

df.text=df.text.astype(str)

test.text=test.text.astype(str)



list_classes = ['num_true', 'num_false', 'num_half_true', 'num_mostly_true','num_barely_true','num_pants_fire'] # six new columns created above

y_t = df[list_classes].values

y_te = test[list_classes].values

max_features = 1000



# tokenize the contents of column "text" from training data

list_sentences_train = df['text']

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)



# tokenize the contents of column "text" from test data

list_sentences_test = test['text']

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_test))

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)



maxlen = 200

# apply pad sequence to tokenized contents

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
# This function was created with help of kernel https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge

def loadEmbeddingMatrix(typeToLoad):

        if(typeToLoad=="word2vec"):

            word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)

            embed_size = 300



        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):

            embeddings_index = dict()

            #Transfer the embedding weights into a dictionary by iterating through every line of the file.

            f = open(EMBEDDING_FILE)

            for line in f:

                #split up line into an indexed array

                values = line.split()

                #first index is word

                word = values[0]

                #store the rest of the values in the array as a new array

                coefs = np.asarray(values[1:], dtype='float32')

                embeddings_index[word] = coefs #50 dimensions

            f.close()

            print('Loaded %s word vectors.' % len(embeddings_index))

        else:

            embeddings_index = dict()

            for word in word2vecDict.wv.vocab:

                embeddings_index[word] = word2vecDict.word_vec(word)

            print('Loaded %s word vectors.' % len(embeddings_index))

            

        gc.collect()

        #We get the mean and standard deviation of the embedding weights so that we could maintain the 

        #same statistics for the rest of our own random generated weights. 

        all_embs = np.stack(list(embeddings_index.values()))

        emb_mean,emb_std = all_embs.mean(), all_embs.std()

        

        nb_words = len(tokenizer.word_index)

        #We are going to set the embedding size to the pretrained dimension as we are replicating it.

        #the size will be Number of Words in Vocab X Embedding Size

        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        gc.collect()



        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 

        #our own dictionary and loaded pretrained embedding. 

        embeddedCount = 0

        for word, i in tokenizer.word_index.items():

            i-=1

            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights

            embedding_vector = embeddings_index.get(word)

            #and store inside the embedding matrix that we will train later on.

            if embedding_vector is not None: 

                embedding_matrix[i] = embedding_vector

                embeddedCount+=1

        print('total embedded:',embeddedCount,'common words')

        

        del(embeddings_index)

        gc.collect()

        

        #finally, return the embedding matrix

        return embedding_matrix
embedding_matrix = loadEmbeddingMatrix('word2vec')
# check the shape of embedding_matrix

embedding_matrix.shape
inp = Input(shape=(maxlen,)) #maxlen=200 as defined earlier

x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

x = Bidirectional(LSTM(120, return_sequences=True,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.2)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(6, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # set the optimizer



# compile the model

model.compile(loss='binary_crossentropy',

                  optimizer=sgd,

                  metrics=['accuracy'])
# Looks for the architecture of model

model.summary()
batch_size = 32

epochs = 30

# Fit the model on training data

hist = model.fit(X_t,y_t, batch_size=batch_size, epochs=epochs, validation_data=(X_te,y_te), shuffle=True, verbose=0)
acc = hist.history['acc']

val_acc = hist.history['val_acc']

loss = hist.history['loss']

val_loss = hist.history['val_loss']



# print the accuracy and loss

print("Loss: %.3f" % min(loss))

print("Validation Loss: %.3f" % min(val_loss))

print("Accuracy: %.3f" % max(acc))

print("Validation Accuracy: %.3f" % max(val_acc))
y1 = df.label

y2 = test1.label



# extract the words from the data using Count Vectorizer

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(list_sentences_train)

count_test = count_vectorizer.transform(list_sentences_test)



# extract the words from the data using Tfidf Vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(list_sentences_train)

tfidf_test = tfidf_vectorizer.transform(list_sentences_test)
mn_count_clf = MultinomialNB(alpha=0.9)



mn_count_clf.fit(count_train, y1)

pred = mn_count_clf.predict(count_test)

score = metrics.accuracy_score(y2, pred)

print("accuracy:   %0.3f" % score)
mn_tfidf_clf = MultinomialNB(alpha=0.9)



mn_tfidf_clf.fit(tfidf_train, y1)

pred = mn_tfidf_clf.predict(tfidf_test)

score = metrics.accuracy_score(y2, pred)

print("accuracy:   %0.3f" % score)
pa_tfidf_clf = PassiveAggressiveClassifier()



pa_tfidf_clf.fit(tfidf_train, y1)

pred = pa_tfidf_clf.predict(tfidf_test)

score = metrics.accuracy_score(y2, pred)

print("accuracy:   %0.3f" % score)
pa_tfidf_clf = PassiveAggressiveClassifier()



pa_tfidf_clf.fit(count_train, y1)

pred = pa_tfidf_clf.predict(count_test)

score = metrics.accuracy_score(y2, pred)

print("accuracy:   %0.3f" % score)
svc_tfidf_clf = LinearSVC()

svc_tfidf_clf.fit(tfidf_train, y1)

pred = svc_tfidf_clf.predict(tfidf_test)

score = metrics.accuracy_score(y2, pred)

print("accuracy:   %0.3f" % score)
sgd_tfidf_clf = SGDClassifier()



sgd_tfidf_clf.fit(tfidf_train, y1)

pred = sgd_tfidf_clf.predict(tfidf_test)

score = metrics.accuracy_score(y2, pred)

print("accuracy:   %0.3f" % score)