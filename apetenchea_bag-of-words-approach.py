from glob import glob

import pandas as pd

import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

import itertools

import re

import os

import string

import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split

from IPython.display import Image

from math import ceil
def parse_folder(name):

    data = []

    for verdict in ('neg', 'pos'):

        for file in glob(os.path.join(name, verdict, '*.txt')):

            data.append({

                'text': open(file, encoding='utf8').read(),

                'verdict': verdict == 'pos'

            })

    return pd.DataFrame(data)
df_train = parse_folder('../input/aclimdb/aclImdb/train/')

df_test = parse_folder('../input/aclimdb/aclImdb/test/')
df_train.iloc[0].text, df_train.iloc[0].verdict
reviews = ['This movie is good', 'The movie is bad', 'Bad this movie was']
vocabulary = set()

for review in reviews:

    for word in review.split(' '):

        vocabulary.add(word)

print(vocabulary)
doc_term = []

for document in reviews:

    row = {'!document': document}

    row.update({word: document.split(' ').count(word) for word in vocabulary})

    doc_term.append(row)

doc_term = pd.DataFrame(doc_term)

display(doc_term)
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’\'“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize_re(text):

    return re_tok.sub(r' \1 ', text).split()
def tokenize_nltk(text):

    return list(itertools.chain.from_iterable(word_tokenize(sentence) for sentence in sent_tokenize(text)))    
stemmer = PorterStemmer()

def tokenize(text):

    return [stemmer.stem(word) for word in tokenize_nltk(text)]
vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=tokenize, max_features=1000000)
train_doc_term = vectorizer.fit_transform(df_train.text)

test_doc_term = vectorizer.transform(df_test.text)

train_doc_term
Image(url='https://ibin.co/4hxqDwhJnxCE.png')
classes = np.array([1, 0, 0])

doc_term_mat = doc_term.drop('!document', axis='columns').values

display(doc_term_mat)

p_c = np.array([(classes == 0).mean(), (classes == 1).mean()])

p_dc = np.ones((2, doc_term_mat.shape[1])) # use ones because by default every term can appear once in every class

for col in range(doc_term_mat.shape[1]):

    for row in range(doc_term_mat.shape[0]):

        p_dc[classes[doc_term_mat[row][col]]][col] += doc_term_mat[row][col]

for c in (0, 1):

    p_dc[c] = p_dc[c] / p_dc[c].sum()

display(p_c, p_dc)
clf = MultinomialNB()

clf.fit(train_doc_term, df_train.verdict)
clf.score(test_doc_term, df_test.verdict)
weights = np.log(p_dc[1] / p_dc[0])

bias = np.log(p_c[1] / p_c[0])

display(weights, bias, doc_term_mat @ weights + bias)
train_doc_term_bool = train_doc_term > 0

r_neg = (train_doc_term_bool[df_train.verdict.values == 0].sum(0) + 1) / (sum(df_train.verdict == 0) + 1)

r_pos = (train_doc_term_bool[df_train.verdict.values == 1].sum(0) + 1) / (sum(df_train.verdict == 1) + 1)

coef = np.log((r_pos / r_neg).A.flatten())
lreg = LogisticRegression(C=0.2, solver='liblinear', max_iter=500, dual=True) # C comes from regularization

lreg.fit(train_doc_term, df_train.verdict)

lreg.score(test_doc_term, df_test.verdict)
def batch_generator(X, y, batch_size, shuffle=False):

    number_of_batches = ceil(X.shape[0]/batch_size)

    counter = 0

    sample_index = np.arange(X.shape[0])

    if shuffle:

        np.random.shuffle(sample_index)

    while True:

        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        X_batch = X[batch_index,:].toarray()

        y_batch = y[batch_index]

        counter += 1

        yield X_batch, y_batch

        if (counter == number_of_batches):

            if shuffle:

                np.random.shuffle(sample_index)

            counter = 0
net = Sequential([

    Dense(1, activation='sigmoid', input_dim=train_doc_term.shape[1], kernel_regularizer=l2(0.1)),

])

net.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

callbacks = [

    ModelCheckpoint('nn_best.h5', monitor='val_acc', verbose=0, save_weights_only=False, save_best_only=True, mode='max'),

    EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

]
net.set_weights([coef.reshape(-1, 1)])
X_train, X_valid, Y_train, Y_valid = train_test_split(train_doc_term, df_train.verdict.values, test_size=0.2, stratify=df_train.verdict.values)
batch_size = 64

net.fit_generator(

    generator=batch_generator(X_train, Y_train, batch_size, shuffle=True),

    validation_data=batch_generator(X_valid, Y_valid, batch_size), validation_steps=ceil(len(Y_valid) / batch_size),

    epochs=5, steps_per_epoch=ceil(len(Y_train) / batch_size), callbacks=callbacks)
net = load_model('nn_best.h5')
net.evaluate_generator(batch_generator(test_doc_term, df_test.verdict, batch_size=batch_size), steps=ceil(len(df_test.verdict) / batch_size))