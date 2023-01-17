# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

import numpy as np

import pandas as pd

import re

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Input, concatenate, Dropout

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model

from keras import optimizers

from keras.utils import to_categorical

import os

from imblearn.ensemble import BalanceCascade
MAX_TIME = 30

VOCAB_SIZE = 20000

QUES_CLEANING_PATTERN = re.compile("[\s\n\r\t.,:;\-_\'\"?!#&()\/%\[\]\{\}\<\>\\$@\!\*\+\=]")

LSTM_DIM = 256

LSTM_DIMS = [512,256]

NUM_FILTERS = [100,50,5]

FILTER_LENGTHS = [1,2,3,4,5]

DROPOUT = 0.3

LEARNING_RATE = 0.005

NUM_EPOCHS = 8

BATCH_SIZE = 2000

NUM_UNDERSAMPLE = 3

RUS_RATIO = 1.0/4.0
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data = np.array(train_data)

test_data = np.array(test_data)
ct_0 = 0

c = 0

for d in train_data:

    if d[2] == 0:

        ct_0 += 1

    if len(QUES_CLEANING_PATTERN.split(d[1])) > MAX_TIME:

        c+=1

print(ct_0, len(train_data) - ct_0, c/len(train_data))
#Preprocess

NUM_TRAIN = int(len(train_data)*0.8)

np.random.shuffle(train_data)

train_x = train_data[:NUM_TRAIN,1]

train_y = train_data[:NUM_TRAIN,2]

val_x = train_data[NUM_TRAIN:,1]

val_y = train_data[NUM_TRAIN:,2]

del train_data
test_x = test_data[:,1]

test_ids = test_data[:,0]

del test_data
tokenizer = Tokenizer(VOCAB_SIZE)

tokenizer.fit_on_texts(train_x)
train_x = tokenizer.texts_to_sequences(train_x)

val_x = tokenizer.texts_to_sequences(val_x)

test_x = tokenizer.texts_to_sequences(test_x)
train_y.shape
def getEmbeddingMatrix(wordIndex):

    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,

       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".

    Input:

        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser

    Output:

        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding

    """

    embeddingsIndex = {}

    # Load the embedding vectors from ther GloVe file

    with open("../input/embeddings/glove.840B.300d/glove.840B.300d.txt", encoding="utf8") as f:

        for line in f:

            values = line.split(' ')

            word = values[0]

            try:

                embeddingVector = np.asarray(values[1:], dtype='float32')

            except:

                print(values)

                break

            embeddingsIndex[word] = embeddingVector



    print('Found %s word vectors.' % len(embeddingsIndex))



    # Minimum word index of any word is 1.

    embeddingMatrix = np.zeros((len(wordIndex) + 1, 300))

    for word, i in wordIndex.items():

        embeddingVector = embeddingsIndex.get(word)

        if embeddingVector is not None:

            # words not found in embedding index will be all-zeros.

            embeddingMatrix[i] = embeddingVector

    del embeddingsIndex

    return embeddingMatrix
wordIndex = tokenizer.word_index

wI = {}

for k,v in wordIndex.items():

    if v < VOCAB_SIZE:

        wI[k] = v

wordIndex = wI
embeddingMatrix = getEmbeddingMatrix(wordIndex)
train_x = pad_sequences(train_x,maxlen=MAX_TIME,padding='post')

val_x = pad_sequences(val_x,maxlen=MAX_TIME,padding='post')

test_x = pad_sequences(test_x,maxlen=MAX_TIME,padding='post')
train_y = to_categorical(train_y)

#train_y = train_y.astype('int')

val_y = to_categorical(val_y)
train_y.shape
def buildModel(embeddingMatrix):

    embeddingLayer = Embedding(embeddingMatrix.shape[0],300,weights=[embeddingMatrix],\

                              input_length=MAX_TIME,trainable=False)

    model = Sequential()

    model.add(embeddingLayer)

    model.add(LSTM(LSTM_DIM,dropout=DROPOUT))

    #model.add(LSTM(LSTM_DIMS[0],dropout = DROPOUT,return_sequences=True))

    #model.add(LSTM(LSTM_DIMS[1],dropout=DROPOUT))

    model.add(Dense(32,activation='relu'))

    model.add(Dense(2,activation='sigmoid'))

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)

    

    model.compile(loss='categorical_crossentropy',

                 optimizer=rmsprop,

                 metrics=['acc'])

    return model
def buildCNNModel(embeddingMatrix):

    sent = Input(shape=(MAX_TIME,))

    embeddingLayer = Embedding(embeddingMatrix.shape[0],300,weights=[embeddingMatrix],\

                              input_length=MAX_TIME,trainable=False)

    sent_emb = embeddingLayer(sent)

    

    feats = []

    for fil_len in FILTER_LENGTHS:

        out = Conv1D(NUM_FILTERS[0],fil_len)(sent_emb)

        out = Dropout(DROPOUT)(out)

        out = BatchNormalization()(out)

        for fil_len2 in FILTER_LENGTHS:

            out2 = Conv1D(NUM_FILTERS[1],fil_len2)(out)

            out2 = Dropout(DROPOUT)(out2)

            out2 = BatchNormalization()(out2)

            for fil_len3 in FILTER_LENGTHS:

                out3 = Conv1D(NUM_FILTERS[2],fil_len3)(out2)

                out3 = Dropout(DROPOUT)(out3)

                out3 = BatchNormalization()(out3)

                feats.append(GlobalMaxPooling1D()(out3))

    

    print(len(feats),feats[0].shape)

    feats = concatenate(feats,axis=-1)

    

    probs = Dense(len(FILTER_LENGTHS)*len(FILTER_LENGTHS),activation = 'relu')(feats)

    probs = Dense(2,activation='sigmoid')(probs)

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)

    

    model = Model(inputs=sent,outputs=probs)

    model.compile(loss='categorical_crossentropy',

                 optimizer=rmsprop,

                 metrics=['acc'])

    return model
#model = buildModel(embeddingMatrix)

model = buildCNNModel(embeddingMatrix)
#rus = BalanceCascade(RUS_RATIO,random_state=i)

#train_x_res, train_y_res = rus.fit_resample(train_x,train_y)

#print(train_y_res.shape)

#train_y_res = to_categorical(train_y_res)

#print(train_y_res.shape)

#perm = np.random.permutation(len(train_y_res))

#train_x_res = train_x_res[perm]

#train_y_res = train_y_res[perm]



#print(train_x_res.shape,train_y_res.shape)

#break

model.fit(train_x,train_y,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=(val_x,val_y))
predictions = model.predict(test_x,batch_size=BATCH_SIZE)

predictions = predictions.argmax(axis=1)
np.sum(predictions==1)/len(test_x)
np.save("preds.npy",predictions)
import csv

with open('submission.csv','w') as f:

    writer = csv.writer(f)

    writer.writerow(['qid','prediction'])

    for i,idx in enumerate(test_ids):

        writer.writerow([idx,predictions[i]])
val_pred = model.predict(val_x,batch_size=BATCH_SIZE)

val_pred = val_pred.argmax(axis=1)
from sklearn.metrics import precision_recall_fscore_support as fscore
fs = fscore(val_y.argmax(axis=1),val_pred)

fs