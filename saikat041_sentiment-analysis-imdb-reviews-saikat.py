import numpy as np

import keras

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.preprocessing import sequence

from keras.layers.embeddings import Embedding

from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM, Dropout

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.layers.convolutional import MaxPooling1D



import time

from sklearn.metrics import accuracy_score 

from matplotlib import pyplot

from scipy import stats

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
def prepareData(top_words):

    vocab = {}

    train_pos = open('../input/train-pos.txt','r').read().split('\n')[0:12500]

    train_neg = open('../input/train-neg.txt','r').read().split('\n')[0:12500]

    

    test_pos = open('../input/test-pos.txt','r').read().split('\n')[0:12500]

    test_neg = open('../input/test-neg.txt','r').read().split('\n')[0:12500]

    

    train = train_pos + train_neg

    test = test_pos + test_neg

    

    for sent in train:

        for word in sent.split():

            if word in vocab:

                vocab[word]+=1

            else:

                vocab[word]=1

    vocab_arr = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

    vocab_arr = [x[0] for x in vocab_arr]

    top_vocab = {word:index for index,word in enumerate(vocab_arr[0:top_words])}  # index 0 reserved for padding and OOV words

    

    x_train = []

    y_train = []

    

    x_test = []

    y_test = []

    

    for index,sent in enumerate(train):

        t = []

        for word in sent.split():

            if word in top_vocab:

                t.append(top_vocab[word])

            else:

                t.append(0)

                

        x_train.append(t)

        

        if index<12500:

            y_train.append(1)

        else:

            y_train.append(0)



    for index,sent in enumerate(test):

        t = []

        for word in sent.split():

            if word in top_vocab:

                t.append(top_vocab[word])

            else:

                t.append(0)

                

        x_test.append(t)

        

        if index<12500:

            y_test.append(1)

        else:

            y_test.append(0)

            

    return (x_train,y_train), (x_test, y_test)



def padSequences(arr, max_words):        

    ans = [x[0:max_words] if len(x)>=max_words else [x[i] if i<len(x) else 0 for i in range(max_words)] for x in arr]

    

    return ans



def changeToNP(x_train, y_train, x_test, y_test):

    y_train = np.array(y_train)

    y_test = np.array(y_test)

    

    np_x_train = np.array([np.array(x) for x in x_train])

    np_y_train = np.array(y_train)

    

    np_x_test = np.array([np.array(x) for x in x_test])

    np_y_test = np.array(y_test)

    

    return (np_x_train, np_y_train), (np_x_test, np_y_test)



def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



def accuracy(y_test, pred):

    pred = [1 if item>=0.5 else 0 for item in pred]

    return accuracy_score(y_test, pred)*100

    

def ensemble(models,x_test,y_test, ensemble_type):

    

    preds = np.zeros(shape = (3,len(x_test)), dtype=np.float32)

    

    

    for index,model in enumerate(models):

        pred = model.predict(x_test).flatten()

        preds[index] = pred

    

    # for mean

    if ensemble_type == 'mean':

        pred = preds.mean(axis=0)

        return accuracy(y_test, pred)

        



    # for mode

    for i in range(preds.shape[0]):

        for j in range(preds.shape[1]):

            if preds[i][j]>=0.5:

                preds[i][j]=1

            else:

                preds[i][j]=0



    pred = stats.mode(preds)[0][0]

        

    return accuracy(y_test, pred)
# for averaging

def train():

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dim, input_length=max_words))

    model.add(Flatten())

    model.add(Dense(250, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])                              #keras model for ANN

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=0)

    scores = model.evaluate(x_test, y_test, verbose=0)



    annAcc = scores[1]*100

    annModel = model

    



                                              

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dim, input_length=max_words))



    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) 

    model.add(MaxPooling1D(pool_size=2))



    model.add(Flatten())                                                                                            #keras model for CNN

    model.add(Dense(250, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=0)

    scores = model.evaluate(x_test, y_test, verbose=0)

    cnnAcc = scores[1]*100

    cnnModel = model

    



    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dim))

    model.add(Bidirectional(LSTM(32, return_sequences = True)))

    model.add(GlobalMaxPool1D())    

    model.add(Dense(32, activation="relu"))                                                                        #keras model RNN                          

    # model.add(Dropout(0.2)) 

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train,  validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=0)

    scores = model.evaluate(x_test, y_test, verbose=0)

    rnnAcc = scores[1]*100

    rnnModel = model

    

    ensemAccMean = ensemble([annModel,cnnModel, rnnModel], x_test, y_test, 'mean')

    ensemAccMode = ensemble([annModel,cnnModel, rnnModel], x_test, y_test, 'mode')

    

    return annAcc, cnnAcc, rnnAcc , ensemAccMean, ensemAccMode
max_words = 500

vocab_size = 5000

embedding_dim = 32



# my dataset

(x_train, y_train), (x_test, y_test) = prepareData(vocab_size)

x_train = padSequences(x_train, max_words)

x_test = padSequences(x_test, max_words)

(x_train, y_train), (x_test, y_test) = changeToNP(x_train, y_train, x_test, y_test)



# keras dataset

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# x_train = sequence.pad_sequences(x_train, maxlen=max_words)

# x_test = sequence.pad_sequences(x_test, maxlen=max_words)
# print("Training ANN Model")

# model = Sequential()

# model.add(Embedding(vocab_size, embedding_dim, input_length=max_words))

# model.add(Flatten())

# model.add(Dense(250, activation='relu'))

# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])                              #keras model for ANN

# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)

# scores = model.evaluate(x_test, y_test, verbose=0)



# print("Accuracy: %.2f%%" % (scores[1]*100))

# pyplot.figure(1)

# pyplot.plot(history.history['acc'], label='train')

# pyplot.plot(history.history['val_acc'], label='test')

# annModel = model
# print("Training CNN Model")                                                

# model = Sequential()

# model.add(Embedding(vocab_size, embedding_dim, input_length=max_words))



# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) 

# model.add(MaxPooling1D(pool_size=2))



# model.add(Flatten())                                                                                            #keras model for CNN

# model.add(Dense(250, activation='relu'))

# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)

# scores = model.evaluate(x_test, y_test, verbose=0)

# print("Accuracy: %.2f%%" % (scores[1]*100))

# pyplot.figure(2)

# pyplot.plot(history.history['acc'], label='train')

# pyplot.plot(history.history['val_acc'], label='test')

# cnnModel = model
# print("Training RNN Model")

# model = Sequential()

# model.add(Embedding(vocab_size, embedding_dim))

# model.add(Bidirectional(LSTM(32, return_sequences = True)))

# model.add(GlobalMaxPool1D())    

# model.add(Dense(32, activation="relu"))                                                                        #keras model RNN                          

# # model.add(Dropout(0.2)) 

# model.add(Dense(1, activation="sigmoid"))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(x_train, y_train,  validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)

# scores = model.evaluate(x_test, y_test, verbose=0)

# print("Accuracy: %.2f%%" % (scores[1]*100))

# pyplot.figure(3)

# pyplot.plot(history.history['acc'], label='train')

# pyplot.plot(history.history['val_acc'], label='test')

# rnnModel = model
# print("Mean Ensemble testing of ANN CNN RNN ")

# acc = ensemble([annModel,cnnModel, rnnModel], x_test, y_test, 'mean')

# print ('Accuracy : %.2f%%'% (acc))

# print("Mode Ensemble testing of ANN CNN RNN ")

# acc = ensemble([annModel,cnnModel, rnnModel], x_test, y_test, 'mode')

# print ('Accuracy : %.2f%%'% (acc))
tot_ann_ac, tot_cnn_ac, tot_rnn_ac , tot_ensem_ac_mean, tot_ensem_ac_mode = 0,0,0,0,0

ans = []

n=5

for i in range(n):

    ann_ac, cnn_ac, rnn_ac , ensem_ac_mean, ensem_ac_mode = train()

    ans.append([ann_ac, cnn_ac, rnn_ac , ensem_ac_mean, ensem_ac_mode])

    tot_ann_ac+=ann_ac

    tot_cnn_ac+=cnn_ac

    tot_rnn_ac+=rnn_ac

    tot_ensem_ac_mean+=ensem_ac_mean

    tot_ensem_ac_mode+=ensem_ac_mode



print("ANN : ", tot_ann_ac/n)    

print("CNN : ", tot_cnn_ac/n)    

print("RNN : ", tot_rnn_ac/n)    

print("ENSEM_MEAN : ", tot_ensem_ac_mean/n) 

print("ENSEM_MODE : ", tot_ensem_ac_mode/n)   

print(np.matrix(ans))