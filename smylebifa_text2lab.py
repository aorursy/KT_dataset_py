from sklearn.metrics import  confusion_matrix

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.metrics import f1_score

from keras.models import Sequential, load_model

from keras.layers import Dense, LSTM, Embedding, Dropout

from keras.layers import Flatten

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences 

import torch as torch

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

print(os.listdir("../input"))

path = './../input/aclimdb/'
def read(path,folder):

    imdb_path = os.path.join(path, 'aclImdb')

    data,labels=[],[]

    for label in ['pos', 'neg']:

            f_path = os.path.join(imdb_path, folder, label)

            for file in sorted(os.listdir(f_path)):

                with open(os.path.join(f_path,file),'rb')as f:

                    review=f.read().decode('utf-8').replace('\n','')

                    data.append(review)

                    labels.append(1 if label== 'pos' else 0)

    return data,labels
X_train_data,y_train_data=read(path,'train')

X_test_data,y_test_data=read(path,'test')
tokenizer = Tokenizer(num_words=5000,split=" ") 

tokenizer.fit_on_texts(X_train_data) 

X_train_en = tokenizer.texts_to_sequences(X_train_data) 

X_test_en =tokenizer.texts_to_sequences(X_test_data)

X_train_new = pad_sequences(X_train_en, maxlen=500) 

X_test_new = pad_sequences(X_test_en, maxlen=500)
model = Sequential()

model.add(Embedding(5000, 20, input_length=500))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary() 

batch_size = 128

epochs =3

model.fit(X_train_new, y_train_data, epochs=epochs, batch_size=batch_size, verbose=2,validation_split=0.2)

predictions = model.predict(X_test_new)

y_pred_data = (predictions > 0.5)

confusion_matrix = confusion_matrix(y_pred_data,y_test_data)

print('accuracy: {0}'.format(f1_score(y_pred_data,y_test_data)))

pd.DataFrame(confusion_matrix)
def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs ={

        'ngram_range' : (1, 2),

        'dtype' : 'int32',

        'strip_accents' : 'unicode',

        'decode_error' : 'replace',

        'analyzer' : 'word',

        'min_df' : 2,

    }

    

    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    x_train = tfidf_vectorizer.fit_transform(train_texts)

    x_val = tfidf_vectorizer.transform(val_texts)

    

    selector = SelectKBest(f_classif, k=min(6000, x_train.shape[1]))

    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float32')

    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val
train, test = ngram_vectorize(X_train_data,y_train_data,X_test_data)
nb = MultinomialNB()

nb.fit(train, y_train_data)

pred = nb.predict(test)

print('accuracy: {0}'.format(f1_score(y_test_data, pred)))