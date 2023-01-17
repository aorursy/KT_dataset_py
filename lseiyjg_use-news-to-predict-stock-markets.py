import re

import nltk

import pandas as pd

import numpy as np



from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

english_stemmer=nltk.stem.SnowballStemmer('english')



from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import random

import itertools



import sys

import os

import argparse

from sklearn.pipeline import Pipeline

from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer

import six

from abc import ABCMeta

from scipy import sparse

from scipy.sparse import issparse

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils import check_X_y, check_array

from sklearn.utils.extmath import safe_sparse_dot

from sklearn.preprocessing import normalize, binarize, LabelBinarizer

from sklearn.svm import LinearSVC



from keras.preprocessing import sequence

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Lambda

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM, SimpleRNN, GRU

from keras.preprocessing.text import Tokenizer

from collections import defaultdict

from keras.layers.convolutional import Convolution1D

from keras import backend as K

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline

plt.style.use('ggplot')
data = pd.read_csv('../input/Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']

test = data[data['Date'] > '2014-12-31']
trainheadlines = []

for row in range(0,len(train.index)):

    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
basicvectorizer = CountVectorizer()

basictrain = basicvectorizer.fit_transform(trainheadlines)

print(basictrain.shape)
basicmodel = LogisticRegression()

basicmodel = basicmodel.fit(basictrain, train["Label"])
testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

basictest = basicvectorizer.transform(testheadlines)

preds1 = basicmodel.predict(basictest)

acc1=accuracy_score(test['Label'], preds1)
print('Logic Regression 1 accuracy: ',acc1 )
basicwords = basicvectorizer.get_feature_names()

basiccoeffs = basicmodel.coef_.tolist()[0]

coeffdf = pd.DataFrame({'Word' : basicwords, 

                        'Coefficient' : basiccoeffs})

coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])

coeffdf.head(5)
coeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.97, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = LogisticRegression()

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds2 = advancedmodel.predict(advancedtest)

acc2=accuracy_score(test['Label'], preds2)
print('Logic Regression 2 accuracy: ', acc2)
advwords = advancedvectorizer.get_feature_names()

advcoeffs = advancedmodel.coef_.tolist()[0]

advcoeffdf = pd.DataFrame({'Words' : advwords, 

                        'Coefficient' : advcoeffs})

advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])

advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.0039, max_df=0.1, max_features = 200000, ngram_range = (3, 3))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = LogisticRegression()

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds3 = advancedmodel.predict(advancedtest)

acc3 = accuracy_score(test['Label'], preds3)
print('Logic Regression 3 accuracy: ', acc3)
advwords = advancedvectorizer.get_feature_names()

advcoeffs = advancedmodel.coef_.tolist()[0]

advcoeffdf = pd.DataFrame({'Words' : advwords, 

                        'Coefficient' : advcoeffs})

advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])

advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 1))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = MultinomialNB(alpha=0.01)

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds4 = advancedmodel.predict(advancedtest)

acc4=accuracy_score(test['Label'], preds4)
print('NBayes 1 accuracy: ', acc4)
advwords = advancedvectorizer.get_feature_names()

advcoeffs = advancedmodel.coef_.tolist()[0]

advcoeffdf = pd.DataFrame({'Words' : advwords, 

                        'Coefficient' : advcoeffs})

advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])

advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = MultinomialNB(alpha=0.0001)

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds5 = advancedmodel.predict(advancedtest)

acc5 = accuracy_score(test['Label'], preds5)
print('NBayes 2 accuracy: ', acc5)
advwords = advancedvectorizer.get_feature_names()

advcoeffs = advancedmodel.coef_.tolist()[0]

advcoeffdf = pd.DataFrame({'Words' : advwords, 

                        'Coefficient' : advcoeffs})

advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])

advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.01, max_df=0.99, max_features = 200000, ngram_range = (1, 1))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = RandomForestClassifier()

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds6 = advancedmodel.predict(advancedtest)

acc6 = accuracy_score(test['Label'], preds6)
print('RF 1 accuracy: ', acc6)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = RandomForestClassifier()

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds7 = advancedmodel.predict(advancedtest)

acc7 = accuracy_score(test['Label'], preds7)
print('RF 2 accuracy: ', acc7)
advancedvectorizer = TfidfVectorizer( min_df=0.1, max_df=0.9, max_features = 200000, ngram_range = (1, 1))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = GradientBoostingClassifier()

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds8 = advancedmodel.predict(advancedtest.toarray())

acc8 = accuracy_score(test['Label'], preds8)
print('GBM 1 accuracy: ', acc8)
advancedvectorizer = TfidfVectorizer( min_df=0.02, max_df=0.175, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = GradientBoostingClassifier()

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds9 = advancedmodel.predict(advancedtest.toarray())

acc9 = accuracy_score(test['Label'], preds9)
print('GBM 2 accuracy: ', acc9)
advancedvectorizer = TfidfVectorizer( min_df=0.2, max_df=0.8, max_features = 200000, ngram_range = (1, 1))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds10 = advancedmodel.predict(advancedtest.toarray())

acc10 = accuracy_score(test['Label'], preds10)
print('SGDClassifier 1: ', acc10)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds11 = advancedmodel.predict(advancedtest.toarray())

acc11 = accuracy_score(test['Label'], preds11)
print('SGDClassifier 2: ', acc11)
class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):



    def __init__(self, alpha=1.0, C=1.0, max_iter=10000):

        self.alpha = alpha

        self.max_iter = max_iter

        self.C = C

        self.svm_ = [] # fuggly



    def fit(self, X, y):

        X, y = check_X_y(X, y, 'csr')

        _, n_features = X.shape



        labelbin = LabelBinarizer()

        Y = labelbin.fit_transform(y)

        self.classes_ = labelbin.classes_

        if Y.shape[1] == 1:

            Y = np.concatenate((1 - Y, Y), axis=1)



        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.

        # so we don't have to cast X to floating point

        Y = Y.astype(np.float64)



        # Count raw events from data

        n_effective_classes = Y.shape[1]

        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)

        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,

                                 dtype=np.float64)

        self._compute_ratios(X, Y)



        # flugglyness

        for i in range(n_effective_classes):

            X_i = X.multiply(self.ratios_[i])

            svm = LinearSVC(C=self.C, max_iter=self.max_iter)

            Y_i = Y[:,i]

            svm.fit(X_i, Y_i)

            self.svm_.append(svm) 



        return self



    def predict(self, X):

        n_effective_classes = self.class_count_.shape[0]

        n_examples = X.shape[0]



        D = np.zeros((n_effective_classes, n_examples))



        for i in range(n_effective_classes):

            X_i = X.multiply(self.ratios_[i])

            D[i] = self.svm_[i].decision_function(X_i)

        

        return self.classes_[np.argmax(D, axis=0)]

        

    def _compute_ratios(self, X, Y):

        """Count feature occurrences and compute ratios."""

        if np.any((X.data if issparse(X) else X) < 0):

            raise ValueError("Input X must be non-negative")



        self.ratios_ += safe_sparse_dot(Y.T, X)  # ratio + feature_occurrance_c

        normalize(self.ratios_, norm='l1', axis=1, copy=False)

        row_calc = lambda r: np.log(np.divide(r, (1 - r)))

        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)

        check_array(self.ratios_)

        self.ratios_ = sparse.csr_matrix(self.ratios_)



        #p_c /= np.linalg.norm(p_c, ord=1)

        #ratios[c] = np.log(p_c / (1 - p_c))





def f1_class(pred, truth, class_val):

    n = len(truth)



    truth_class = 0

    pred_class = 0

    tp = 0



    for ii in range(0, n):

        if truth[ii] == class_val:

            truth_class += 1

            if truth[ii] == pred[ii]:

                tp += 1

                pred_class += 1

                continue;

        if pred[ii] == class_val:

            pred_class += 1



    precision = tp / float(pred_class)

    recall = tp / float(truth_class)



    return (2.0 * precision * recall) / (precision + recall)





def semeval_senti_f1(pred, truth, pos=2, neg=0): 



    f1_pos = f1_class(pred, truth, pos)

    f1_neg = f1_class(pred, truth, neg)



    return (f1_pos + f1_neg) / 2.0;





def main(train_file, test_file, ngram=(1, 3)):

    print('loading...')

    train = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,

                        names=['text', 'label'])



    # to shuffle:

    #train.iloc[np.random.permutation(len(df))]



    test = pd.read_csv(test_file, delimiter='\t', encoding='utf-8', header=0,

                        names=['text', 'label'])



    print('vectorizing...')

    vect = CountVectorizer()

    classifier = NBSVM()



    # create pipeline

    clf = Pipeline([('vect', vect), ('nbsvm', classifier)])

    params = {

        'vect__token_pattern': r"\S+",

        'vect__ngram_range': ngram, 

        'vect__binary': True

    }

    clf.set_params(**params)



    #X_train = vect.fit_transform(train['text'])

    #X_test = vect.transform(test['text'])



    print('fitting...')

    clf.fit(train['text'], train['label'])



    print('classifying...')

    pred = clf.predict(test['text'])

   

    print('testing...')

    acc = accuracy_score(test['label'], pred)

    f1 = semeval_senti_f1(pred, test['label'])

    print('NBSVM: acc=%f, f1=%f' % (acc, f1))
advancedvectorizer = TfidfVectorizer( min_df=0.1, max_df=0.8, max_features = 200000, ngram_range = (1, 1))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)

print(advancedtrain.shape)
advancedmodel = NBSVM(C=0.01)

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds12 = advancedmodel.predict(advancedtest)

acc12 = accuracy_score(test['Label'], preds12)
print('NBSVM 1: ', acc12)
advancedvectorizer = TfidfVectorizer( min_df=0.031, max_df=0.2, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)

print(advancedtrain.shape)
advancedmodel = NBSVM(C=0.01)

advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

preds13 = advancedmodel.predict(advancedtest)

acc13 = accuracy_score(test['Label'], preds13)
print('NBSVM 2: ', acc13)
batch_size = 32

nb_classes = 2

advancedvectorizer = TfidfVectorizer( min_df=0.04, max_df=0.3, max_features = 200000, ngram_range = (2, 2))

advancedtrain = advancedvectorizer.fit_transform(trainheadlines)

testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

advancedtest = advancedvectorizer.transform(testheadlines)

print(advancedtrain.shape)
X_train = advancedtrain.toarray()

X_test = advancedtest.toarray()



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)

y_train = np.array(train["Label"])

y_test = np.array(test["Label"])



Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)





# pre-processing: divide by max and substract mean

scale = np.max(X_train)

X_train /= scale

X_test /= scale



mean = np.mean(X_train)

X_train -= mean

X_test -= mean



input_dim = X_train.shape[1]



# Here's a Deep Dumb MLP (DDMLP)

model = Sequential()

model.add(Dense(256, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



print("Training...")

model.fit(X_train, Y_train, nb_epoch=2, batch_size=16, validation_split=0.15, show_accuracy=True)



print("Generating test predictions...")

preds14 = model.predict_classes(X_test, verbose=0)

acc14 = accuracy_score(test["Label"], preds14)
print('prediction accuracy: ', acc14)
max_features = 10000

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.1

maxlen = 200

batch_size = 32

nb_classes = 2
# vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(nb_words=max_features)

tokenizer.fit_on_texts(trainheadlines)

sequences_train = tokenizer.texts_to_sequences(trainheadlines)

sequences_test = tokenizer.texts_to_sequences(testheadlines)
print('Pad sequences (samples x time)')

X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)

X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)



Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)





print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)
print('Build model...')

model = Sequential()

model.add(Embedding(max_features, 128, dropout=0.2))

model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 

model.add(Dense(nb_classes))

model.add(Activation('softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print('Train...')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=3,

          validation_data=(X_test, Y_test))

score, acc = model.evaluate(X_test, Y_test,

                            batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)





print("Generating test predictions...")

preds15 = model.predict_classes(X_test, verbose=0)

acc15 = accuracy_score(test['Label'], preds15)
print('prediction accuracy: ', acc15)
nb_filter = 120

filter_length = 2

hidden_dims = 120

nb_epoch = 2
print('Build model...')

model = Sequential()

model.add(Embedding(max_features, 128, dropout=0.2))

# we add a Convolution1D, which will learn nb_filter

# word group filters of size filter_length:

model.add(Convolution1D(nb_filter=nb_filter,

                        filter_length=filter_length,

                        border_mode='valid',

                        activation='relu',

                        subsample_length=1))



def max_1d(X):

    return K.max(X, axis=1)



model.add(Lambda(max_1d, output_shape=(nb_filter,)))

model.add(Dense(hidden_dims)) 

model.add(Dropout(0.2)) 

model.add(Activation('relu'))

model.add(Dense(nb_classes))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
print('Train...')

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1,

          validation_data=(X_test, Y_test))

score, acc = model.evaluate(X_test, Y_test,

                            batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)





print("Generating test predictions...")

preds16 = model.predict_classes(X_test, verbose=0)

acc16 = accuracy_score(test['Label'], preds16)
print('prediction accuracy: ', acc16)