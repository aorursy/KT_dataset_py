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

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier, SGDRegressor

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
#https://www.kaggle.com/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones
def review_to_wordlist( review, remove_stopwords=True):

    # Function to convert a document to a sequence of words,

    # optionally removing stop words.  Returns a list of words.

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(review,"lxml").get_text()



    #

    # 2. Remove non-letters

    review_text = re.sub("[^a-zA-Z]"," ", review)

    #

    # 3. Convert words to lower case and split them

    words = review_text.lower().split()

    #

    # 4. Optionally remove stop words (True by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]



    b=[]

    stemmer = english_stemmer #PorterStemmer()

    for word in words:

        b.append(stemmer.stem(word))



    # 5. Return a list of words

    return(b)
data_file = '../input/Amazon_Unlocked_Mobile.csv'



n = 413000  

s = 20000 

skip = sorted(random.sample(range(1,n),n-s))





data = pd.read_csv( data_file, delimiter = ",", skiprows = skip)
data.shape
print (data[1:3])
data = data[data['Reviews'].isnull()==False]
train, test = train_test_split(data, test_size = 0.3)
sns.countplot(data['Rating'])
clean_train_reviews = []

for review in train['Reviews']:

    clean_train_reviews.append( " ".join(review_to_wordlist(review)))

    

clean_test_reviews = []

for review in test['Reviews']:

    clean_test_reviews.append( " ".join(review_to_wordlist(review)))
vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),

                              sublinear_tf = True )



vectorizer = vectorizer.fit(clean_train_reviews)

train_features = vectorizer.transform(clean_train_reviews)



test_features = vectorizer.transform(clean_test_reviews)
fselect = SelectKBest(chi2 , k=10000)

train_features = fselect.fit_transform(train_features, train["Rating"])

test_features = fselect.transform(test_features)
#print (train_features[1:4])

print (test_features[1:4])
model1 = MultinomialNB(alpha=0.001)

model1.fit( train_features, train["Rating"] )



model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)

model2.fit( train_features, train["Rating"] )



model3 = RandomForestClassifier()

model3.fit( train_features, train["Rating"] )



model4 = GradientBoostingClassifier()

model4.fit( train_features, train["Rating"] )



pred_1 = model1.predict( test_features.toarray() )

pred_2 = model2.predict( test_features.toarray() )

pred_3 = model3.predict( test_features.toarray() )

pred_4 = model4.predict( test_features.toarray() )
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
model5 = NBSVM(C=0.01)

model5.fit( train_features, train["Rating"] )



pred_5 = model5.predict( test_features )
print(classification_report(test['Rating'], pred_2, target_names=['1','2','3','4','5']))
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Compute confusion matrix

cnf_matrix = confusion_matrix(test['Rating'], pred_5)
plot_confusion_matrix(cnf_matrix, classes=['1','2','3','4','5'],

                      title='Confusion matrix, without normalization')
print('prediction 1 accuracy: ', accuracy_score(test['Rating'], pred_1))

print('prediction 2 accuracy: ', accuracy_score(test['Rating'], pred_2))

print('prediction 3 accuracy: ', accuracy_score(test['Rating'], pred_3))

print('prediction 4 accuracy: ', accuracy_score(test['Rating'], pred_4))

print('prediction 5 accuracy: ', accuracy_score(test['Rating'], pred_5))
batch_size = 32

nb_classes = 5
vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 1000, ngram_range = ( 1, 3 ),

                              sublinear_tf = True )



vectorizer = vectorizer.fit(clean_train_reviews)

train_features = vectorizer.transform(clean_train_reviews)



test_features = vectorizer.transform(clean_test_reviews)
X_train = train_features.toarray()

X_test = test_features.toarray()



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)

y_train = np.array(train['Rating']-1)

y_test = np.array(test['Rating']-1)



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

model.add(Dropout(0.2))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



print("Training...")

model.fit(X_train, Y_train, nb_epoch=5, batch_size=16, validation_split=0.1, show_accuracy=True)



print("Generating test predictions...")

preds = model.predict_classes(X_test, verbose=0)
print('prediction 6 accuracy: ', accuracy_score(test['Rating'], preds+1))
max_features = 20000

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2

maxlen = 80

batch_size = 32

nb_classes = 5
# vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(nb_words=max_features)

tokenizer.fit_on_texts(train['Reviews'])

sequences_train = tokenizer.texts_to_sequences(train['Reviews'])

sequences_test = tokenizer.texts_to_sequences(test['Reviews'])
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

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,

          validation_data=(X_test, Y_test))

score, acc = model.evaluate(X_test, Y_test,

                            batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)





print("Generating test predictions...")

preds = model.predict_classes(X_test, verbose=0)
print('prediction 7 accuracy: ', accuracy_score(test['Rating'], preds+1))
nb_filter = 250

filter_length = 3

hidden_dims = 250

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

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,

          validation_data=(X_test, Y_test))

score, acc = model.evaluate(X_test, Y_test,

                            batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)





print("Generating test predictions...")

preds = model.predict_classes(X_test, verbose=0)
print('prediction 8 accuracy: ', accuracy_score(test['Rating'], preds+1))