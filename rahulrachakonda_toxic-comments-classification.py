import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()

train.isnull().any(),test.isnull().any()

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]

max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
list_tokenized_train[:1]

maxlen = 200

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])

plt.show()
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

embed_size = 128

x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
batch_size = 32

epochs = 1

model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.summary()

from keras import backend as K



# with a Sequential model

get_3rd_layer_output = K.function([model.layers[0].input],

                                  [model.layers[2].output])

layer_output = get_3rd_layer_output([X_t[:1]])[0]

layer_output.shape

#print layer_output to see the actual data
def toxicity_level(string):

    """

    Return toxicity probability based on inputed string.

    """

    # Process string

    new_string = [string]

    new_string = tokenizer.texts_to_sequences(new_string)

    new_string = pad_sequences(new_string, maxlen=200, padding='post', truncating='post')

    

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
toxicity_level('Hello, How are you?')

def loss(y_true, y_pred):

     return keras.backend.binary_crossentropy(y_true, y_pred)



lr = .0001

model.compile(loss=loss, optimizer=Nadam(lr=lr, clipnorm=1.0),

              metrics=['binary_accuracy'])
graph = model.fit(X, y, batch_size=batch_size, epochs=epochs,

                  validation_data=(X_val, y_val), callbacks=[RocAuc, early_stop],

                  verbose=1, shuffle=False)



import matplotlib.pyplot as plt

%matplotlib inline



# Visualize history of loss

plt.plot(graph.history['loss'])

plt.plot(graph.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()