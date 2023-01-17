import keras

import csv

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

Train= pd.read_csv("../input/train.tsv", sep="\t")

Test = pd.read_csv("../input/test.tsv", sep="\t")



Train.shape
Train.head()
Train['Phrase'].str.len().mean()

Train['Phrase'].str.len().max()

Train['Sentiment'].value_counts()

#Create df of full sentences

fullSent = Train.loc[Train.groupby('SentenceId')['PhraseId'].idxmin()]



#Change sentiment to increase readability

fullSent['sentiment_label'] = ''

Sentiment_Label = ['Negative', 'Somewhat Negative', 

                  'Neutral', 'Somewhat Positive', 'Positive']

for sent, label in enumerate(Sentiment_Label):

    fullSent.loc[Train.Sentiment == sent, 'sentiment_label'] = label

    

fullSent.head()
#Add non-helpful stopwords to stopword list

Stopwords = list(ENGLISH_STOP_WORDS)

Stopwords.extend(['movie','movies','film','nt','rrb','lrb',

                      'make','work','like','story','time','little'])



#Create tfidf vectorizer object & fit to full sentence training data

tfidf_vectorizor = TfidfVectorizer(min_df=5, 

                             max_df=0.5,

                             analyzer='word',

                             strip_accents='unicode',

                             ngram_range=(1, 3),

                             sublinear_tf=True, 

                             smooth_idf=True,

                             use_idf=True,

                             stop_words=Stopwords)



tfidf_vectorizor.fit(list(fullSent['Phrase']))
BoW_vectorizer = CountVectorizer(strip_accents='unicode',

                                 stop_words=Stopwords,

                                 ngram_range=(1,3),

                                 analyzer='word',

                                 min_df=5,

                                 max_df=0.5)



BoW_vectorizer.fit(list(fullSent['Phrase']))
phrase = np.array(Train['Phrase'])

sentiment = np.array(Train['Sentiment'])

# build train and test datasets

phrase_train, phrase_test, sentiment_train, sentiment_test = train_test_split(phrase, 

                                                                              sentiment, 

                                                                              test_size=0.2, 

                                                                              random_state=4)



#TF-IDF

train_tfidfmatrix = tfidf_vectorizor.fit_transform(phrase_train)

test_tfidfmatrix = tfidf_vectorizor.transform(phrase_test)



#Vectorizer (Bag of Words Model)

train_simplevector = BoW_vectorizer.transform(phrase_train)

test_simplevector = BoW_vectorizer.transform(phrase_test)
def format_data(train, test, max_features, maxlen):

    

    from keras.preprocessing.text import Tokenizer

    from keras.preprocessing.sequence import pad_sequences

    from keras.utils import to_categorical

    

    train = train.sample(frac=1).reset_index(drop=True)

    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())

    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())



    X = train['Phrase']

    test_X = test['Phrase']

    Y = to_categorical(train['Sentiment'].values)



    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(X))



    X = tokenizer.texts_to_sequences(X)

    X = pad_sequences(X, maxlen=maxlen)

    test_X = tokenizer.texts_to_sequences(test_X)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    return X, Y, test_X
maxlen = 125

max_features = 10000



X, Y, test_X = format_data(Train, Test, max_features, maxlen)
X
Y
test_X
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=4)
from keras import backend as K

def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall

      

def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        precision = true_positives / (possible_positives + K.epsilon())

        return precision

      

        

def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Activation, GlobalMaxPooling1D

from keras.layers import SpatialDropout1D

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.models import Sequential

from keras import optimizers

def baseline_cnn_model(fea_matrix, n_class, mode, compiler):

    model = Sequential()



# Input / Embdedding

    model.add(Embedding(max_features, 150, input_length=maxlen))



# CNN

    model.add(SpatialDropout1D(0.2))



    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=2))



    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=2))



    model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=2))



    model.add(Flatten())

    model.add(Dense(5, activation='relu'))

    if n_class == 1 and mode == "cla":

        #model.add(Activation('sigmoid'))

        model.compile(optimizer=compiler, loss='binary_crossentropy',

                  metrics=['acc', f1_m,precision_m,recall_m])

    else:

        #model.add(Activation('softmax'))

        model.compile(optimizer=compiler, loss='categorical_crossentropy',

                  metrics=['acc', f1_m,precision_m,recall_m])

    return model

# Output layer

    #model.add(Dense(5, activation='sigmoid'))
epochs = 100

#batch_size = 32

lr = 1e-4

batch_size = 100

#num_epochs = 10

decay=1e-2

mode = "reg"

n_class = 3 #5



adm = optimizers.Adam(lr = lr, decay = decay)

sgd = optimizers.SGD(lr = lr, nesterov=True, momentum=0.7, decay=decay)

Nadam = optimizers.Nadam(lr = lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model = baseline_cnn_model(X_train, n_class, mode, Nadam)

#if n_class == 1 and mode == "cla":

#        model.add(Activation('sigmoid'))

#        model.compile(optimizer='adam', loss='binary_crossentropy',

#                  metrics=['acc', f1_m,precision_m,recall_m])

#else:

#        model.add(Activation('softmax'))

#        model.compile(optimizer='adam', loss='categorical_crossentropy',

#                  metrics=['acc', f1_m,precision_m,recall_m])



#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)
def print_metrics(accuracy, f1_score, precision, recall):

    print("Simple cnn model performance")

    print('Accuracy: ', np.round(accuracy, 4))

    print('Precision: ', np.round(precision, 4))

    print('Recall: ', np.round(recall, 4))

    print('F1 score: ', np.round(f1_score, 4))

    print('\n')

    

loss, accuracy, f1_score, precision, recall = model.evaluate(X, Y)

print_metrics(accuracy, f1_score, precision, recall)


