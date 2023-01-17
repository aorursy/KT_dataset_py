!pip install contractions



import pandas as pd

import re

import contractions

import en_core_web_sm

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras import Sequential

from keras.layers import Embedding, Dense, Dropout, GlobalMaxPool1D

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
#LOAD DATA

o = pd.read_csv("../input/onion-or-not/OnionOrNot.csv")



#SHOW FIVE ROWS

o.head(5)
#FIX CONTRACTIONS

o['text'] = o['text'].apply(lambda x: contractions.fix(x))



#REMOVE PUNCTUATION

o['text'] = o['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))



#CONVERT TO LOWERCASE



def lowerCase(input_str):

    input_str = input_str.lower()

    return input_str



o['text'] = o['text'].apply(lambda x: lowerCase(x))



#SHOW FIVE ROWS

o.head(5)
#LEMMATIZATION

sp = en_core_web_sm.load()



def lemma(input_str):

    s = sp(input_str)

    

    input_list = []

    for word in s:

        w = word.lemma_

        input_list.append(w)

        

    output = ' '.join(input_list)

    return output



o['text'] = o['text'].apply(lambda x: lemma(x))



#SHOW FIVE ROWS

o.head(5)
#VECTORIZE

tokenizer = Tokenizer(num_words = 10000, split = ' ')

tokenizer.fit_on_texts(o['text'].values)



X = tokenizer.texts_to_sequences(o['text'].values)

X = pad_sequences(X)



y = o['label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,

                                                    random_state = 42)
#BUILD THE MODEL

model = Sequential()



model.add(Embedding(10000, 128, input_length = X.shape[1]))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(GlobalMaxPool1D())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',

              metrics=['accuracy'])



model.summary()
#TRAIN THE MODEL

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)



history = model.fit(X_train, y_train, 

                    epochs = 1000, batch_size = 32, verbose = 0, #YOU CAN CHANGE verbose = 1 TO SEE THE PROCESS

                    validation_data = (X_test, y_test), callbacks=[es])
#SHOW ACCURACY AND CONFUSION MATRIX

y_pred = model.predict(X_test)

y_pred = y_pred > 0.5



accuracy_score(y_pred, y_test)
confusion_matrix(y_pred, y_test)