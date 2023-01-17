# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding,Activation,Flatten,merge,TimeDistributed,CuDNNGRU,Bidirectional

from keras.layers.merge import concatenate

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import concatenate,subtract,add,maximum,multiply,Layer,Lambda

from keras.backend import backend as K



from tqdm import tqdm

## read data from path

MAX_SEQUENCE_LENGTH = 40

MAX_NB_WORDS = 4000000

nrows = 10000000

EMBEDDING_DIM = 300



# nrows = 10000



path = '../input/quora-question-pairs/'

train = pd.read_csv(path+"train.csv",nrows=nrows).astype(str)

test = pd.read_csv(path+"test.csv",nrows=nrows).astype(str)
train.head(10)
## generate keras inputs

## tokenizer and padding

corpus = []



feats = ['question1','question2']

for f in feats:

    train[f] = train[f].astype(str)

    test[f] = test[f].astype(str)

    corpus+=train[f].values.tolist()

    

    

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(corpus)

X_q1 = tokenizer.texts_to_sequences(train['question1'])

X_q2 = tokenizer.texts_to_sequences(train['question2'])



X_test_q1 = tokenizer.texts_to_sequences(test['question1'])

X_test_q2 = tokenizer.texts_to_sequences(test['question2'])





X_q1 = pad_sequences(X_q1, maxlen=MAX_SEQUENCE_LENGTH)

X_q2 = pad_sequences(X_q2, maxlen=MAX_SEQUENCE_LENGTH)

X_test_q1 = pad_sequences(X_test_q1, maxlen=MAX_SEQUENCE_LENGTH)

X_test_q2 = pad_sequences(X_test_q2, maxlen=MAX_SEQUENCE_LENGTH)



y = train['is_duplicate'].values



word_index = tokenizer.word_index

print('Found %s unique tokens' % len(word_index))

nb_words = min(MAX_NB_WORDS, len(word_index))+1

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words



glove_path = '../input/glove840b300dtxt/glove.840B.300d.txt'

embedding_matrix,unknown_words = build_matrix(word_index,glove_path)
## eval metric is logloss

from sklearn.metrics import log_loss
## validation set

from sklearn.model_selection import train_test_split

X_train_q1,X_val_q1,X_train_q2,X_val_q2,y_train,y_val = train_test_split(X_q1,X_q2,y,train_size=0.8,random_state=1024)

print(X_train_q1.shape,X_val_q1.shape)

X_train = [X_train_q1,X_train_q2]

X_val = [X_val_q1,X_val_q2]

X_test = [X_test_q1,X_test_q2]

def get_model_siamese(nb_words=50000,MAX_SEQUENCE_LENGTH=200,EMBEDDING_DIM=200,act='relu',embedding_matrix=None):

    ########################################

    ## define the model structure

    ########################################

    

    input_q1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    input_q2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



    if embedding_matrix is not None:

        weights=[embedding_matrix],



        embedding_layer = Embedding(nb_words,

                EMBEDDING_DIM,

                input_length=MAX_SEQUENCE_LENGTH,

                weights=[embedding_matrix],

                trainable=False)

    else:

        embedding_layer = Embedding(nb_words,

                EMBEDDING_DIM,

                input_length=MAX_SEQUENCE_LENGTH,

                trainable=True)



    embedded_sequences_q1 = embedding_layer(input_q1)

    embedded_sequences_q2 = embedding_layer(input_q2)



    

    bilstm_layer = Bidirectional(CuDNNGRU(64,return_sequences=False))



    x1 = bilstm_layer(embedded_sequences_q1)

    x2 = bilstm_layer(embedded_sequences_q2)



    diff = subtract([x1,x2])

    summation = add([x1,x2])

    

    x = concatenate([x1,x2,diff,summation],1)



    x = Dense(64,activation=act)(x)

    

    preds = Dense(1,activation='sigmoid')(x)





    model = Model(inputs=[input_q1,input_q2],outputs=preds)

    model.compile(loss='binary_crossentropy',

            optimizer='adam',

            metrics=['accuracy'])

    print(model.summary())

    

    return model
## train model



early_stopping =EarlyStopping(monitor='val_loss', patience=5)

bst_model_path = 'best_model.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)



model = get_model_siamese(nb_words=nb_words,MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,EMBEDDING_DIM=EMBEDDING_DIM,act='relu',embedding_matrix=embedding_matrix)

model.fit(X_train,y_train,epochs=250,validation_data=(X_val, y_val),batch_size=128,shuffle=True,verbose=1,callbacks=[early_stopping, model_checkpoint])



model.load_weights(bst_model_path)

## make submission 

y_pred_test = model.predict(X_test)



submission = pd.DataFrame()

submission['test_id'] = test['test_id']

submission['is_duplicate'] = y_pred_test

submission.to_csv('submission_siamese.csv',index=False)
