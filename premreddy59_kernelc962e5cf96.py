lr = 1e-3

batch_size = 128

num_epochs = 100

decay = 1e-4

mode = "reg"

n_class = 5 #5

import csv

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn import metrics

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
URL_Tr ='https://raw.githubusercontent.com/cacoderquan/Sentiment-Analysis-on-the-Rotten-Tomatoes-movie-review-dataset/master/train.tsv'

URL_Te ='https://raw.githubusercontent.com/cacoderquan/Sentiment-Analysis-on-the-Rotten-Tomatoes-movie-review-dataset/master/test.tsv'
train = pd.read_csv(URL_Tr,sep='\t')

test = pd.read_csv(URL_Te,sep='\t')

train.head()

test.head()
print(train.shape,"\n",test.shape)
print("\t",train.isnull().values.any(), "\n\t",

      test.isnull().values.any()

     )
#sanitization

fullSent = train.loc[train.groupby('SentenceId')['PhraseId'].idxmin()]



fullSent.head()
print (len(train.groupby('SentenceId').nunique()),

      len(test.groupby('SentenceId').nunique())

      )
StopWords = ENGLISH_STOP_WORDS

print(StopWords)
BOW_Vectorizer = CountVectorizer(strip_accents='unicode',

                                 stop_words=StopWords,

                                 ngram_range=(1,3),

                                 analyzer='word',

                                 min_df=5,

                                 max_df=0.5)



BOW_Vectorizer.fit(list(fullSent['Phrase']))
#create tfidf vectorizer 

tfidf_vectorizer = TfidfVectorizer(min_df=5,

                                 max_df=5,

                                  analyzer='word',

                                  strip_accents='unicode',

                                  ngram_range=(1,3),

                                  sublinear_tf=True,

                                  smooth_idf=True,

                                  use_idf=True,

                                  stop_words=StopWords)



tfidf_vectorizer.fit(list(fullSent['Phrase']))
#tfid

#build train and test datasets

phrase = fullSent['Phrase']

sentiment = fullSent['Sentiment']

phrase[0], sentiment[0]
X_train,X_test,Y_train,Y_test = train_test_split(phrase,sentiment,test_size=0.2,random_state=4)



X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
#calling both the methods

#method 1-BOW

train_bow=BOW_Vectorizer.transform(X_train)

test_bow=BOW_Vectorizer.transform(X_test)

train_bow.shape[1]

bow_feature_vec = pd.DataFrame(train_bow.toarray(), columns = BOW_Vectorizer.get_feature_names())

bow_feature_vec.head(15)



bow_feature_vec_test = pd.DataFrame(test_bow.toarray(), columns = BOW_Vectorizer.get_feature_names())

bow_feature_vec_test.head(15)
from keras import backend as K

def recall_m(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

  recall = true_positives / (possible_positives + K.epsilon())

  return recall



def precision_m(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

  predicted_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

  precision = true_positives / (predicted_positives + K.epsilon())

  return precision



def f1_m(y_true, y_pred):

  precision = precision_m(y_true, y_pred)

  recall = recall_m(y_true, y_pred)

  return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

from keras.layers import Activation, Conv1D, GlobalMaxPooling1D

from keras import optimizers
fea_vec_dim = bow_feature_vec.shape[1]

print(fea_vec_dim, n_class)



X_train = bow_feature_vec.values.reshape((bow_feature_vec.shape[0], bow_feature_vec.shape[1], 1))

X_train.shape







fea_vec_test_dim = bow_feature_vec_test.shape[1]

print(fea_vec_test_dim, n_class)



X_test = bow_feature_vec_test.values.reshape((bow_feature_vec_test.shape[0], bow_feature_vec_test.shape[1], 1))

X_test.shape



def baseline_cnn_model(fea_matrix, n_class, mode, compiler):

  #create model

  model = Sequential()

  model.add(Conv1D(filters=64, kernel_size = 3, activation = 'relu',

                  input_shape=(fea_matrix.shape[1], fea_matrix.shape[2])))

  model.add(MaxPooling1D(pool_size = 2))

  model.add(Conv1D(filters=128, kernel_size = 3, activation = 'relu'))

  model.add(MaxPooling1D(pool_size=2))

  model.add(Flatten())

  model.add(Activation('relu'))

  model.add(Dense(n_class))

  if n_class==1 and mode == "cla":

    model.add(Activation('sigmoid'))

    # compile the model

    model.compile(optimizer=compiler, loss = 'binary_crossentropy',

                 metrics=['acc', f1_m, precision_m, recall_m])

  else:

    model.add(Activation('softmax'))  

    #comoile the model

    model.compile(optimizer=compiler, loss = 'sparse_categorical_crossentropy',

                 metrics=['acc', f1_m, precision_m, recall_m])

  return model

  




adm = optimizers.Adam(lr = lr, decay = decay)

sgd = optimizers.SGD(lr = lr, nesterov = True, momentum = 0.7, decay = decay)

Nadam = optimizers.Nadam(lr = lr, beta_1=0.9, beta_2=0.999, epsilon = 1e-08)

model = baseline_cnn_model(X_train, n_class, mode, Nadam)
model.fit(X_train, Y_train, batch_size = batch_size, 

          epochs = num_epochs, verbose=1, validation_split = 0.2)
def print_metrics(accuracy, f1_score, precision, recall):

  print('SIMPLE CNN MODEL PERFORMANCE')

  print('Accuracy: ', np.round(accuracy, 4))

  print('Precision: ', np.round(precision, 4))

  print('Recall: ', np.round(recall, 4))

  print('F1 Score: ', np.round(f1_score, 4))

  print('\n')
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test)

print_metrics(accuracy, f1_score, precision, recall)