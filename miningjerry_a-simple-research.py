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
# import 

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
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/train_book.csv")

data.head()
train_book = pd.read_csv("../input/train_book.csv")

train_dvd = pd.read_csv("../input/train_dvd.csv")

train_music = pd.read_csv("../input/train_music.csv")



test_book = pd.read_csv("../input/testResult_book_trans.csv")

test_dvd = pd.read_csv("../input/testResult_dvd_trans.csv")

test_music = pd.read_csv("../input/testResult_music_trans.csv")



test_book.columns = ["Comment","Polarity"]

test_dvd.columns = ["Comment","Polarity"]

test_music.columns = ["Comment","Polarity"]



print("train_book:{0}, with columns{1}".format(len(train_book), train_book.columns))

print("train_dvd:{0}, with columns{1}".format(len(train_dvd), train_dvd.columns))

print("train_music:{0}, with columns{1}".format(len(train_music), train_music.columns))

print("test_book:{0}, with columns{1}".format(len(test_book), test_book.columns))

print("test_dvd:{0}, with columns{1}".format(len(test_dvd), test_dvd.columns))

print("test_music:{0}, with columns{1}".format(len(test_music), test_music.columns))

def combine(data):

    data_text = data["Summary"] + ". "+ data["Summary.1"]+". "+data["Comment"]

    print(data_text.shape)

    return data_text
def review_to_wordlist( review, remove_stopwords=True):

    # Function to convert a document to a sequence of words,

    # optionally removing stop words.  Returns a list of words.

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(review).get_text()



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
def preprocess(train, test):

    train_text = combine(train)

    clean_train_reviews = []

    for review in train_text:

        clean_train_reviews.append( " ".join(review_to_wordlist(review)))



    clean_test_reviews = []

    for review in test['Comment']:

        clean_test_reviews.append( " ".join(review_to_wordlist(review)))



    vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),

                                  sublinear_tf = True )



    vectorizer = vectorizer.fit(clean_train_reviews)

    train_features = vectorizer.transform(clean_train_reviews)



    test_features = vectorizer.transform(clean_test_reviews)



    fselect = SelectKBest(chi2 , k=5000)

    train_features = fselect.fit_transform(train_features, train["Polarity"])

    test_features = fselect.transform(test_features)

    

    print ("Feature finished")

    return train_features, test_features
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
def train_model(train, test):

    train_features, test_features = preprocess(train, test)

    

    model1 = MultinomialNB(alpha=0.001)

    model1.fit( train_features, train["Polarity"] )



    #model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)

    #model2.fit( train_features, train["Polarity"] )



    model3 = RandomForestClassifier()

    model3.fit( train_features, train["Polarity"] )



    #model4 = GradientBoostingClassifier()

    #model4.fit( train_features, train["Polarity"] )



    pred_1 = model1.predict( test_features.toarray() )

    #pred_2 = model2.predict( test_features.toarray() )

    pred_3 = model3.predict( test_features.toarray() )

    #pred_4 = model4.predict( test_features.toarray() )

    

    model5 = NBSVM(C=0.01)

    model5.fit( train_features, train["Polarity"] )



    pred_5 = model5.predict( test_features )

    

    print("Prediction finished, evalueing...")

    print(classification_report(test['Polarity'], pred_1, target_names=['1','2']))

    #print(classification_report(test['Polarity'], pred_2, target_names=['1','2']))

    print(classification_report(test['Polarity'], pred_3, target_names=['1','2']))

    #print(classification_report(test['Polarity'], pred_4, target_names=['1','2']))

    print(classification_report(test['Polarity'], pred_5, target_names=['1','2']))

    return pred_1, pred_3, pred_5
print("book:")

pred_1, pred_3, pred_5=train_model(train_book, test_book)


pred_11 = [-1 if i =='N' else 1 for i in pred_1 ]

pred_31 = [-1 if i =='N' else 1 for i in pred_3 ]

pred_51 = [2 if i =='N' else 2 for i in pred_5]



pred = [pred_11[i] +pred_31[i]+pred_51[i] for i,item in enumerate(zip(pred_11,pred_31,pred_51))]



print(pred_1[:10])

print(pred_3[:10])

print(pred_5[:10])

print(pred[:10])
pred_n = ['P' if i>2 else 'N' for i in pred]

print(classification_report(test_book['Polarity'], pred_n, target_names=['1','2'])) 
print(classification_report(test_book['Polarity'], pred_n, target_names=['1','2']))  
result = []

for i in range(10):

    item = s.iloc[i,:]

    print(item.sum(axis=1))







    if item.sum(axis=1) >= 4:

        result.append(1)

    else:

        result.append(0)

print(result)

print(result[:10])

#print(classification_report(test['Polarity'], result, target_names=['1','2']))            


print("book:")

train_model(train_book, test_book)

print("dvd:")

train_model(train_dvd, test_dvd)

print("music:")

train_model(train_music, test_music)
def model_LSTM(train, test,n_size):

    train_text = combine(train)

    y_train = train["Polarity"]

    y_test = test["Polarity"]

    

    y_train = [0 if item == 'N' else 1 for item in y_train]

    y_test = [0 if item == 'N' else 1 for item in y_test]

    

    max_features = 20000

    EMBEDDING_DIM = 100

    VALIDATION_SPLIT = 0.2

    maxlen = 80

    batch_size = n_size

    nb_classes = 2

    print('token........')

    # vectorize the text samples into a 2D integer tensor

    tokenizer = Tokenizer(nb_words=max_features)

    tokenizer.fit_on_texts(train_text)

    sequences_train = tokenizer.texts_to_sequences(train_text)

    sequences_test = tokenizer.texts_to_sequences(test['Comment'])

    

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

    

    print("Prediction finished, evalueing...")

    print(classification_report(y_test, preds, target_names=['1','2']))

    

    print('over........................................')

train_data=pd.concat([train_book, train_dvd, train_music], axis=0)

test_data=pd.concat([test_book, test_dvd, test_music], axis=0)



print(train_data.shape)

print(test_data.shape)
print('aa')
model_LSTM(train_data, test_data,25)
train_data=pd.concat([train_book, train_dvd, train_music], axis=0)

test_data=pd.concat([test_book, test_dvd, test_music], axis=0)



print(train_data.shape)

print(test_data.shape)
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

# cnf_matrix = confusion_matrix(test['Polarity'], pred_5)
from keras.preprocessing.text import Tokenizer  

from keras.preprocessing.sequence import pad_sequences  

from keras.utils.np_utils import to_categorical  

from keras.layers import Dense, Input, Flatten  

from keras.layers import Conv1D, MaxPooling1D, Embedding  

from keras.models import Model  

from keras.optimizers import *  

from keras.models import Sequential  

from keras.layers import Merge  

import sys  

def model_CNN(train, test,n_batch):

    

    max_features = 20000

    #EMBEDDING_DIM = 100

    VALIDATION_SPLIT = 0.2

    maxlen = 80

    batch_size = n_batch

    nb_classes = 2

    

    nb_filter = 250

    filter_length = 3

    hidden_dims = 250

    nb_epoch = 2

    

    X_train = combine(train)

    X_test = test['Comment']

    y_train = train["Polarity"]

    y_test = test["Polarity"]

    

    y_train = [0 if item == 'N' else 1 for item in y_train]

    y_test = [0 if item == 'N' else 1 for item in y_test]

    

    # vectorize the text samples into a 2D integer tensor

    tokenizer = Tokenizer(nb_words=max_features)

    tokenizer.fit_on_texts(X_train)

    sequences_train = tokenizer.texts_to_sequences(X_train)

    sequences_test = tokenizer.texts_to_sequences(X_test)



    print('Pad sequences (samples x time)')

    X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)

    X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)



    Y_train = np_utils.to_categorical(y_train, nb_classes)

    Y_test = np_utils.to_categorical(y_test, nb_classes)

    

    val_size = int(VALIDATION_SPLIT * X_train.shape[0]) 

    x_val = X_train[:val_size]

    y_val = Y_train[:val_size]

    

    X_train = X_train[val_size:]

    Y_train = Y_train[val_size:]



    print('X_train shape:', X_train.shape)

    print('X_test shape:', X_test.shape)



    print('Build model...')

    def max_1d(X):

        return K.max(X, axis=1)

    

    embedding_layer = Embedding(max_features, 128, dropout=0.2) 

    

    print('model1')

    print('*' * 100)

    model_1 = Sequential()

    model_1.add(embedding_layer)

    # we add a Convolution1D, which will learn nb_filter

    # word group filters of size filter_length:

    model_1.add(Convolution1D(nb_filter=nb_filter,

                            filter_length=filter_length,

                            border_mode='valid',

                            activation='relu',

                            subsample_length=1))

    model_1.add(Lambda(max_1d, output_shape=(nb_filter,)))    

    print('model2')

    print('*' * 100)    

    model_2 = Sequential()

    model_2.add(embedding_layer)

    # we add a Convolution1D, which will learn nb_filter

    # word group filters of size filter_length:

    model_2.add(Convolution1D(nb_filter=nb_filter,

                            filter_length=80,

                            border_mode='valid',

                            activation='relu',

                            subsample_length=1))

    model_2.add(Lambda(max_1d, output_shape=(nb_filter,)))    



    print('model3')

    print('*' * 100)    

    model_3 = Sequential()

    model_3.add(embedding_layer)

    # we add a Convolution1D, which will learn nb_filter

    # word group filters of size filter_length:

    model_3.add(Convolution1D(nb_filter=nb_filter,

                            filter_length=100,

                            border_mode='valid',

                            activation='relu',

                            subsample_length=1))

    model_3.add(Lambda(max_1d, output_shape=(nb_filter,)))    



    

    merged = Merge([model_1, model_2,model_3], mode='concat')     

    model = Sequential()  

    model.add(merged) # add merge  

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

                validation_data=(x_val, y_val))

    score, acc = model.evaluate(x_val, y_val,

                                batch_size=batch_size)

    print('Test score:', score)

    print('Test accuracy:', acc)

    

    

    print("Prediction finished, evalueing validation...")

    preds_val = model.predict_classes(x_val, verbose=0)

    print(classification_report(y_val, preds_val, target_names=['1','2']))

    conf_matrix = confusion_matrix(y_val, preds_val)

    print(conf_matrix)



    #print("Generating test predictions...")

    #preds = model.predict_classes(X_test, verbose=0)

    #print("Prediction finished, evalueing...")

    #print(classification_report(y_test, preds, target_names=['1','2']))

    

    print('over........................................')

    return  conf_matrix,preds_val
print('aa')
conf_matrix,preds = model_CNN(train_data, test_data,20)
plot_confusion_matrix(cnf_matrix, classes=['P','N'],

                      title='Confusion matrix, without normalization')
print('aa')
conf_matrix,preds = model_CNN(train_data, test_data,20)



plot_confusion_matrix(cnf_matrix, classes=['P','N'],

                      title='Confusion matrix, without normalization')
print('aa')
print('aa')
from keras.preprocessing.text import Tokenizer  

from keras.preprocessing.sequence import pad_sequences  

from keras.utils.np_utils import to_categorical  

from keras.layers import Dense, Input, Flatten  

from keras.layers import Conv1D, MaxPooling1D, Embedding  

from keras.models import Model  

from keras.optimizers import *  

from keras.models import Sequential  

from keras.layers import Merge  

import sys  



def model_CNN(train, test):

    

    max_features = 20000

    EMBEDDING_DIM = 100

    VALIDATION_SPLIT = 0.2

    maxlen = 80

    batch_size = 30

    nb_classes = 2

    

    nb_filter = 250

    filter_length = 3

    hidden_dims = 250

    nb_epoch = 2

    VALIDATION_SPLIT = 0.4

    

    X_train = combine(train)

    X_test = test['Comment']

    y_train = train["Polarity"]

    y_test = test["Polarity"]

    

    y_train = [0 if item == 'N' else 1 for item in y_train]

    y_test = [0 if item == 'N' else 1 for item in y_test]

    

    nb_words=max_features

    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))  



    

    # vectorize the text samples into a 2D integer tensor

    tokenizer = Tokenizer(nb_words=max_features)

    tokenizer.fit_on_texts(X_train)

    sequences_train = tokenizer.texts_to_sequences(X_train)

    sequences_test = tokenizer.texts_to_sequences(X_test)



    print('Pad sequences (samples x time)')

    X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)

    X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)



    Y_train = np_utils.to_categorical(y_train, nb_classes)

    Y_test = np_utils.to_categorical(y_test, nb_classes)

    

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])  

  

    #x_train = data[:-nb_validation_samples] # 训练集  

    #y_train = labels[:-nb_validation_samples]# 训练集的标签  

    #x_val = data[-nb_validation_samples:] # 测试集，英文原意是验证集  

    #y_val = labels[-nb_validation_samples:] # 测试集的标签  



    print('X_train shape:', X_train.shape)

    print('X_test shape:', X_test.shape)



    print('Build model...')

    

    embedding_layer = Embedding(nb_words + 1,  

                                EMBEDDING_DIM,  

                                input_length=maxlen,  

                                weights=[embedding_matrix],  

                                trainable=True)  

  

    print('Training model.')  

    print ('model1....')

    print('*'*100)

    def max_1d(X):

        return K.max(X, axis=1)

    





    # train a 1D convnet with global maxpoolinnb_wordsg  



    #left model 第一块神经网络，卷积窗口是5*50（50是词向量维度）  

    model_left = Sequential()  

    #model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))  

    model_left.add(embedding_layer)  

    model_left.add(Conv1D(nb_filter, 128, activation='tanh'))  

    model_left.add(MaxPooling1D(128))  

    model_left.add(Conv1D(nb_filter, 128, activation='tanh'))  

    model_left.add(MaxPooling1D(128))  

    model_left.add(Conv1D(nb_filter, 128, activation='tanh'))  

    model_left.add(MaxPooling1D(128))  

    model_left.add(Flatten())  

  

    #right model <span style="font-family: Arial, Helvetica, sans-serif;">第二块神经网络，卷积窗口是4*50</span>  

    print ('model2....')

    print('*'*100)

    model_right = Sequential()  

    model_right.add(embedding_layer)  

    model_right.add(Conv1D(nb_filter, 100, activation='tanh'))

    model_right.add(MaxPooling1D(100))  

    model_right.add(Conv1D(nb_filter, 100, activation='tanh')) 

    model_right.add(MaxPooling1D(100))  

    model_right.add(Conv1D(nb_filter, 100, activation='tanh'))  

    model_right.add(MaxPooling1D(100))  

    model_right.add(Flatten())  

    print ('model3....')

    print('*'*100)

    #third model <span style="font-family: Arial, Helvetica, sans-serif;">第三块神经网络，卷积窗口是6*50</span>  

    model_3 = Sequential()  

    model_3.add(embedding_layer)  

    model_3.add(Conv1D(nb_filter, 80, activation='tanh'))  

    model_3.add(MaxPooling1D(80))  

    model_3.add(Conv1D(nb_filter, 80, activation='tanh'))  

    model_3.add(MaxPooling1D(80))  

    model_3.add(Conv1D(nb_filter, 80, activation='tanh'))  

    model_3.add(MaxPooling1D(1))  

    model_3.add(Flatten())  

  

  

    merged = Merge([model_left, model_right,model_3], mode='concat') # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本



      

    model = Sequential()  

    model.add(merged) # add merge  

    model.add(Dense(128, activation='relu')) # 全连接层  

    model.add(Dense(nb_classes, activation='sigmoid')) # softmax，输出文本属于2种类





    # 优化器我这里用了adadelta，也可以使用其他方法  

    #model.compile(loss='categorical_crossentropy',  

    #              optimizer='Adadelta',  

    #              metrics=['accuracy'])  

  

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])



    print('Train...')

    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1)

    #          validation_data=(X_test, Y_test))

    #score, acc = model.evaluate(X_test, Y_test,

    #                           batch_size=batch_size)

    # =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢  

    #model.fit(x_train, y_train,nb_epoch=1)  

  

    #score = model.evaluate(x_train, y_train, verbose=0) # 评估模型在训练集中的效果，准确率约

    model.fit(X_train, Y_train,  nb_epoch=1)

    #         validation_data=(X_test, Y_test))

    score, acc = model.evaluate(X_test, Y_test,verbose=0)

    

    

    #print('train score:', score[0])  

    #print('train accuracy:', score[1])  

    #score = model.evaluate(x_val, y_val, verbose=0)  # 评估模型在测试集中的效果，准确率约为

    #print('Test score:', score[0])  

    #print('Test accuracy:', score[1])

    print('Test score:', score)

    print('Test accuracy:', acc)





    print("Generating test predictions...")

    preds = model.predict_classes(X_test, verbose=0)

    print("Prediction finished, evalueing...")

    print(classification_report(y_test, preds, target_names=['1','2']))

    

    print('over........................................')
model_CNN(train_data, test_data)