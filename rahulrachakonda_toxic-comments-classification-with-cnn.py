import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU,Conv1D,MaxPooling1D

from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.callbacks import EarlyStopping, ModelCheckpoint

import gc

from sklearn.model_selection import train_test_split

from keras.models import load_model
train = pd.read_csv('../input/train.csv')

submit = pd.read_csv('../input/test.csv')

X_train, X_test, y_train, y_test = train_test_split(train, train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]], test_size = 0.10, random_state = 42)
list_sentences_train = X_train["comment_text"]

list_sentences_test = X_test["comment_text"]

list_sentences_submit = submit["comment_text"]
max_features = 20000

tokenizer = Tokenizer(num_words=max_features,char_level=True)
tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_sentences_test = tokenizer.texts_to_sequences(list_sentences_test)

list_tokenized_submit = tokenizer.texts_to_sequences(list_sentences_submit)
maxlen = 500

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_sentences_test, maxlen=maxlen)

X_sub = pad_sequences(list_tokenized_submit, maxlen=maxlen)
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

plt.hist(totalNumWords)

plt.show()
inp = Input(shape=(maxlen, ))



embed_size = 240

x = Embedding(len(tokenizer.word_index)+1, embed_size)(inp)



x = Conv1D(filters=100,kernel_size=4,padding='same', activation='relu')(x)



x=MaxPooling1D(pool_size=4)(x)



x = Bidirectional(GRU(60, return_sequences=True,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x)



x = GlobalMaxPool1D()(x)



x = Dense(50, activation="relu")(x)



x = Dropout(0.2)(x)

x = Dense(6, activation="sigmoid")(x)



model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                 metrics=['accuracy'])



model.summary()

batch_size = 32

epochs = 1

hist = model.fit(X_t,y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_te,y_test))
def toxicity_level(string):

    """

    Return toxicity probability based on inputed string.

    """

    # Process string

    new_string = [string]

    new_string = tokenizer.texts_to_sequences(new_string)

    new_string = pad_sequences(new_string, maxlen=500, padding='post', truncating='post')

    

    # Predict

    prediction = model.predict(new_string)

    

    # Print output

    print("Toxicity levels for '{}':".format(string))

    print('Toxic:         {:.0%}'.format(prediction[0][0]))

    print('Severe Toxic:  {:.0%}'.format(prediction[0][1]))

    print('Obscene:       {:.0%}'.format(prediction[0][2]))

    print('Threat:        {:.0%}'.format(prediction[0][3]))

    print('Insult:        {:.0%}'.format(prediction[0][4]))

    print('Identity Hate: {:.0%}'.format(prediction[0][5]))

    print()

    

    return



toxicity_level('go jump off a bridge jerk')

toxicity_level('i will kill you')

toxicity_level('have a nice day')

toxicity_level('hola, como estas')

toxicity_level('hola mierda joder')

toxicity_level('fuck off!!')
def print_metrics(accuracy):

    print("Simple cnn model performance")

    print('Accuracy: ', np.round(accuracy, 4))

    print('\n')

    

loss, accuracy = model.evaluate(X_t, y_train)

print_metrics(accuracy)