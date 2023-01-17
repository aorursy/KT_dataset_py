# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/support-ticket-urgency-classification/"))



# Any results you write to the current directory are saved as output.
import os, sys



import numpy as np



from keras.models import Model



from keras.layers import Input, Dense, Flatten

from keras.layers import Conv1D, MaxPooling1D,Dropout

from keras.layers import Embedding



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.utils import to_categorical
train = pd.read_csv("../input/support-ticket-urgency-classification/all_tickets-1551435513304.csv",header=0)
train.head()
train.drop(["ticket_type","category","sub_category1","sub_category2","business_service","impact"],inplace=True,axis=1)
train.drop("title",inplace=True,axis = 1)
train.shape
X = train.body.values

Y = train.urgency.values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2 , random_state = 2, stratify=Y)

# X_train.shape
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=250,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



print("Words used in less priority 1 tickets")

show_wordcloud(train.body[train.urgency == 1])

print("Words used in less priority 2 tickets")

show_wordcloud(train.body[train.urgency == 2])

print("Words used in less priority 3 tickets")

show_wordcloud(train.body[train.urgency == 3])
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# Prepare tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)



word_Index = tokenizer.word_index



vocab_Size = len(word_Index) + 1

print('Found %s unique tokens.' % vocab_Size)
# Prepare tokenizer

tokenizer_1 = Tokenizer()

tokenizer_1.fit_on_texts(X_test)



word_Index_1 = tokenizer_1.word_index



vocab_Size_1 = len(word_Index_1) + 1

print('Found %s unique tokens.' % vocab_Size_1)
# integer encode the documents

sequences = tokenizer.texts_to_sequences(X_train)

print(X_train[1], sequences[1])

print("------------------------#################-------------------------")

sequences_test = tokenizer.texts_to_sequences(X_test)

print(X_test[1], sequences_test[1])

#for i in sequences:

#    print (len(i))
MAX_SEQUENCE_LENGTH = 1000



data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)



print('Shape of data tensor:', data.shape)

print('Shape of data tensor:', data_test.shape)
# split the data into a training set and a test set

X_train = data



X_test = data_test



y_train = y_train



y_test = y_test
Y_train = to_categorical(y_train)

Y_test = to_categorical(y_test)
y_train.shape
PATH_glove = "../input/glove-6b"



embeddings_index = {}

f = open(os.path.join(PATH_glove, 'glove.6B.50d.txt'), encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_Matrix = np.zeros((vocab_Size, 50))

for word, i in word_Index.items():

    embedding_Vector = embeddings_index.get(word)

    if embedding_Vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_Matrix[i] = embedding_Vector



print (embedding_Matrix.shape)
embedding_layer = Embedding(vocab_Size,

                            50,

                            weights=[embedding_Matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(4)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(4)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(4)(x)  # global max pooling

x = Flatten()(x)

x = Dense(100, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(50, activation='relu')(x)

x = Dropout(0.2)(x)

preds = Dense(4, activation='softmax')(x)



model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# summarize the model

print(model.summary())
model.fit(X_train, Y_train, epochs=5,validation_split=0.2,batch_size=100)
Y_pred = model.predict(X_test)

print(Y_pred)
scores =model.evaluate(X_test, Y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# print the confusion matrix

metrics.confusion_matrix(y_test, y_pred)
from keras.datasets import imdb #A utility to load a dataset



from keras.models import Sequential



from keras.layers import Dense, LSTM, Dropout, Embedding,BatchNormalization,Activation,GRU,CuDNNLSTM,CuDNNGRU,Bidirectional

from keras.layers import Conv1D, MaxPooling1D



from keras.preprocessing import sequence #To convert a variable length sentence into a prespecified length

embedding_vector_length = 150



model_LSTM = Sequential()



model_LSTM.add(Embedding(vocab_Size, embedding_vector_length, input_length=MAX_SEQUENCE_LENGTH))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(100))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(BatchNormalization())

model_LSTM.add(Dense(100))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(BatchNormalization())

model_LSTM.add(Activation(activation='relu'))

model_LSTM.add(Dense(50))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(BatchNormalization())

model_LSTM.add(Activation(activation='relu'))

model_LSTM.add(Dense(20))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(BatchNormalization())

model_LSTM.add(Activation(activation='relu'))

model_LSTM.add(Dense(4, activation='softmax'))
model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_LSTM.summary())
model_LSTM.fit(X_train, Y_train, epochs=4, batch_size=256,validation_split=0.2)
scores =model_LSTM.evaluate(X_test, Y_test, verbose=1,batch_size=500)

print("Accuracy: %.2f%%" % (scores[1]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))
Y_pred =model_LSTM.predict(X_test)

print(Y_pred)
embedding_vector_length = 150



model_GRU = Sequential()



model_GRU.add(Embedding(vocab_Size, embedding_vector_length, input_length=MAX_SEQUENCE_LENGTH))

model_GRU.add(Dropout(0.2))

model_GRU.add(GRU(130))

model_GRU.add(Dropout(0.2))

model_GRU.add(BatchNormalization())

model_GRU.add(Dense(100))

model_GRU.add(Dropout(0.2))

model_GRU.add(BatchNormalization())

model_GRU.add(Activation(activation='relu'))

model_GRU.add(Dense(50))

model_GRU.add(Dropout(0.2))

model_GRU.add(BatchNormalization())

model_GRU.add(Activation(activation='relu'))

model_GRU.add(Dense(20))

model_GRU.add(Dropout(0.2))

model_GRU.add(BatchNormalization())

model_GRU.add(Activation(activation='relu'))

model_GRU.add(Dense(4, activation='softmax'))
model_GRU.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_GRU.summary())
model_GRU.fit(X_train, Y_train, epochs=4, batch_size=256,validation_split=0.2)
scores =model_GRU.evaluate(X_test, Y_test, verbose=1,batch_size=500)

print("Accuracy: %.2f%%" % (scores[1]*100))
embedding_vector_length = 300



model_BDLSTM = Sequential()



model_BDLSTM.add(Embedding(vocab_Size, embedding_vector_length, input_length=MAX_SEQUENCE_LENGTH))

#model_BDLSTM.add(Bidirectional(CuDNNGRU(100,return_sequences=True)))

model_BDLSTM.add(Bidirectional(CuDNNGRU(75)))

model_BDLSTM.add(Dropout(0.2))

model_BDLSTM.add(Dense(10))

model_BDLSTM.add(Dropout(0.2))

model_BDLSTM.add(Activation(activation='relu'))

model_BDLSTM.add(Dense(10))

model_BDLSTM.add(Dropout(0.2))

model_BDLSTM.add(BatchNormalization())

model_BDLSTM.add(Activation(activation='relu'))

model_BDLSTM.add(Dense(8))

model_BDLSTM.add(Dropout(0.2))

model_BDLSTM.add(Activation(activation='relu'))

model_BDLSTM.add(Dense(8))

model_BDLSTM.add(Dropout(0.2))

model_BDLSTM.add(BatchNormalization())

model_BDLSTM.add(Activation(activation='relu'))



model_BDLSTM.add(Dense(4, activation='softmax'))
model_BDLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_BDLSTM.summary())
model_BDLSTM.fit(X_train, Y_train, epochs=5, batch_size=256,validation_split=0.2)
scores =model_BDLSTM.evaluate(X_test, Y_test, verbose=1,batch_size=500)

print("Accuracy: %.2f%%" % (scores[1]*100))
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Bidirectional(CuDNNLSTM(128))(embedded_sequences)

x = Dropout(0.2)(x)

x = Dense(32, activation='relu')(x)

x = Dropout(0.2)(x)

preds = Dense(4, activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train, epochs=9,validation_split=0.2,batch_size=128)
scores =model.evaluate(X_test, Y_test, verbose=1,batch_size=500)

print("Accuracy: %.2f%%" % (scores[1]*100))