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
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
pre_train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

pre_train_comment = pre_train['comment_text'].fillna('')

pre_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

pre_test_comment = pre_test['comment_text'].fillna('')
# pre_train = pd.read_csv('../input/processed/process_train.csv')

# pre_train_comment = pre_train['comment_text'].fillna('')

# pre_test = pd.read_csv('../input/processed/process_test.csv')

# pre_test_comment = pre_test['comment_text'].fillna('')
y = pre_train[labels].values
import tensorflow as tf

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,SpatialDropout1D, concatenate

from keras.layers import Bidirectional,Bidirectional, GRU, GlobalAveragePooling1D,GlobalMaxPooling1D,GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.model_selection import train_test_split

import gc
max_features = 30000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(pre_train_comment))

pre_train_comment = tokenizer.texts_to_sequences(pre_train_comment)

pre_test_comment = tokenizer.texts_to_sequences(pre_test_comment)
maxlen = 100

X_t = pad_sequences(pre_train_comment, maxlen=maxlen)

X_te = pad_sequences(pre_test_comment, maxlen=maxlen)
# from gensim.models import KeyedVectors

# embedding_index = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary=True)

from gensim.models import KeyedVectors

embedding_index =  KeyedVectors.load_word2vec_format('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec')
def loadEmbeddingMatrix(pre_vector):

    embed_size = 300

    embeddings_index = dict()

    for word in pre_vector.wv.vocab:

        embeddings_index[word] = pre_vector.word_vec(word)

    print('Loaded %s word vectors.' % len(embeddings_index))

    

    gc.collect()

    all_embs = np.stack(list(embeddings_index.values()))

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    

    nb_words = len(tokenizer.word_index)

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    gc.collect()



    embeddedCount = 0

    for word, i in tokenizer.word_index.items():

        i -= 1

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

            embeddedCount += 1

    print('total embedded:', embeddedCount,'common words')

    

    del(embeddings_index)

    gc.collect()



    return embedding_matrix

    
import pickle

embedding_matrix = loadEmbeddingMatrix(embedding_index)

pickle.dump(embedding_matrix, open("embedding_matrix.pkl", "wb"))

# import pickle

# embedding_matrix = pickle.load(open('../input/weigths/embedding_matrix.pkl', 'rb'))
embedding_matrix.shape
inp = Input(shape=(maxlen, ))

x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

# x = SpatialDropout1D(0.2)(x)

x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

x = Bidirectional(GRU(80, return_sequences=True))(x)

# avg_pool = GlobalAveragePooling1D()(x)

# max_pool = GlobalMaxPooling1D()(x)

# conc = concatenate([avg_pool, max_pool])

# outp = Dense(6, activation="sigmoid")(conc)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation='relu')(x)

x = Dropout(0.1)(x)

x = Dense(6, activation='sigmoid')(x)
from keras.callbacks import EarlyStopping, ModelCheckpoint

checkpoint = ModelCheckpoint('weights_word2vec.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_loss", mode="max", patience=20)

callbacks_list = [checkpoint, early]
model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
model.summary()
batch_size = 32

epochs = 2

# with tf.device('/gpu:0'):

hist = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
model.load_weights('weights_word2vec.best.hdf5')
y_test = model.predict(X_te)
sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")

sample_submission[labels] = y_test



sample_submission.to_csv("submission.csv", index=False)