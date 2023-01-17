import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics

import gensim.models.keyedvectors as word2vec

import gc



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")

test_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

embed_size=0

print(train_df.shape)

print(test_df.shape)
train_df.head(3)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["comment_text"].fillna("_na_").values

val_X = val_df["comment_text"].fillna("_na_").values

test_X = test_df["comment_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_y = train_df[list_classes].values

val_y = val_df[list_classes].values
def loadEmbeddingMatrix(typeToLoad):

        #load different embedding file from Kaggle depending on which embedding 

        #matrix we are going to experiment with

        if(typeToLoad=="glove"):

            EMBEDDING_FILE='../input/glove840b300dtxt/glove.840B.300d.txt'

            embed_size = 25

        elif(typeToLoad=="word2vec"):

            word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin", binary=True)

            embed_size = 300

        elif(typeToLoad=="fasttext"):

            EMBEDDING_FILE='../input/fasttext/wiki.simple.vec'

            embed_size = 300



        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):

            embeddings_index = dict()

            #Transfer the embedding weights into a dictionary by iterating through every line of the file.

            f = open(EMBEDDING_FILE)

            for line in f:

                #split up line into an indexed array

                values = line.split()

                #first index is word

                word = values[0]

                #store the rest of the values in the array as a new array

                coefs = np.asarray(values[1:], dtype='float32')

                embeddings_index[word] = coefs #50 dimensions

            f.close()

            print('Loaded %s word vectors.' % len(embeddings_index))

        else:

            embeddings_index = dict()

            for word in word2vecDict.wv.vocab:

                embeddings_index[word] = word2vecDict.word_vec(word)

            print('Loaded %s word vectors.' % len(embeddings_index))

            

        gc.collect()

        #We get the mean and standard deviation of the embedding weights so that we could maintain the 

        #same statistics for the rest of our own random generated weights. 

        all_embs = np.stack(list(embeddings_index.values()))

        emb_mean,emb_std = all_embs.mean(), all_embs.std()

        

        nb_words = len(tokenizer.word_index)

        #We are going to set the embedding size to the pretrained dimension as we are replicating it.

        #the size will be Number of Words in Vocab X Embedding Size

        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        gc.collect()



        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 

        #our own dictionary and loaded pretrained embedding. 

        embeddedCount = 0

        for word, i in tokenizer.word_index.items():

            i-=1

            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights

            embedding_vector = embeddings_index.get(word)

            #and store inside the embedding matrix that we will train later on.

            if embedding_vector is not None: 

                embedding_matrix[i] = embedding_vector

                embeddedCount+=1

        print('total embedded:',embeddedCount,'common words')

        

        del(embeddings_index)

        gc.collect()

        

        #finally, return the embedding matrix

        return embedding_matrix
embedding_matrix = loadEmbeddingMatrix('word2vec')
embedding_matrix.shape
inp = Input(shape=(maxlen, )) #maxlen=200 By indicating an empty space after comma, we are telling Keras to infer the number automatically.

x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
model.summary()
hist = model.fit(train_X, train_y, batch_size=512, epochs=7, validation_data=(val_X, val_y))
history_dict = hist.history
import matplotlib.pyplot as plt

acc = hist.history['acc']

val_acc = hist.history['val_acc']

loss = hist.history['loss']

val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()
model.save('word2vec.h5')