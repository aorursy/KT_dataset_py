import numpy as np 

import pandas as pd

import gc

import os

print(os.listdir("../input"))

import sys

from tqdm  import tqdm

tqdm.pandas()

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report


training_data_1 = pd.read_csv("../input/bmw_training_set.csv") 

df = training_data_1

df
one_hot = pd.get_dummies(df["Intent"])

df.drop(['Intent'],axis=1,inplace=True)

df = pd.concat([df,one_hot],axis=1)

df.head()
df1 = df

df1
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1["Utterance"].values, df1.drop(["Utterance"],axis=1).values, test_size=0.2, random_state=42)
print(len(np.unique(X_train)))

print(len(np.unique(X_test)))

print(len(np.unique(y_train)))

print(len(np.unique(y_test)))
vocabulary_size = 2000000

tokenizer = Tokenizer(num_words= vocabulary_size)

print(X_train)



tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)

X_train = pad_sequences(sequences, maxlen=50)

sequences = tokenizer.texts_to_sequences(X_test)

X_test = pad_sequences(sequences, maxlen=50)

X_test
model = Sequential()

model.add(Embedding(200000, 100, input_length=50))

model.add(Dropout(0.2))

model.add(Conv1D(64, 5, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))

model.add(Dense(144, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train,

                    batch_size=1024,

                    epochs=1000,

                    verbose=1,

                    validation_split=0.1)
score = model.evaluate(X_test, y_test,

                       batch_size=256, verbose=1)

print('Test accuracy:', score[1])
preds = model.predict(X_test)

preds
print(classification_report(np.argmax(y_test,axis=1),np.argmax(preds,axis=1)))
gc.collect()
def CountFrequency(my_list,licht_converted): 

    licht_converted_1 = np.array(licht_converted).flatten().tolist()

    #print(licht_converted_1)

    tekcht_licht_1 = tokenizer.sequences_to_texts(licht_converted)

    # Creating an empty dictionary  

    freq = {} 

    for items in tekcht_licht_1: 

        freq[items] = tekcht_licht_1.count(items) 

    #print(freq)   

    return freq



def predict_intent_data(utterence_to_process_to_find_intent):

    predict_data_integer = np.argmax(model.predict(pad_sequences(tokenizer.texts_to_sequences(utterence_to_process_to_find_intent), maxlen=50)), axis=1)

    predict_data_integer_1 = np.sort(predict_data_integer)

    licht_converted = []

    for data_1 in predict_data_integer_1:

        licht_converted.append([data_1])

    prediction_uith_frequency = CountFrequency(predict_data_integer_1.tolist(),licht_converted)

    return prediction_uith_frequency





predict_intent_data("cost of car charging")