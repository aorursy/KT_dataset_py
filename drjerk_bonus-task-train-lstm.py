import numpy as np 

import pandas as pd



import os

import sys

import re

import time

import math

import random

import itertools



from itertools import groupby

from operator import itemgetter



from tqdm import tqdm_notebook as tqdm



from tensorflow.keras.preprocessing.sequence import pad_sequences



from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import accuracy_score



import pickle



import gc

gc.enable()
from tensorflow.python.keras.layers import Dense, Input, CuDNNLSTM, Embedding, Activation, CuDNNGRU, Lambda, Add, Bidirectional, GlobalMaxPooling1D, Concatenate, SpatialDropout1D, GlobalAveragePooling1D

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.models import Model

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.callbacks import LearningRateScheduler

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
CATEGORY_CNT = 54

# Размер батча (train и predict)

BATCH_SIZE = 256

PREDICT_BATCH_SIZE = 256

# Кол-во эпох в одном фолде

EPOCHS = 10

#STEPS_PER_EPOCH = 300

STEPS_PER_EPOCH = 3000

VAL_STEPS_PER_EPOCH = 100

# Размерность word2vec и fastext

WORD2VEC_DIM = 200

FASTTEXT_DIM = 200

# Показатель для экспонентального понижения learning rate-а

EXP_DECAY_COEF = 0.75

# Нужно ли выводить логи keras-а и прогресс бар tqdm

VERBOSE = 1

# Размер корзинки для генерации батчей (см сильно внизу)

MAXLEN = 300

#

TRAINABLE_EMBED_SIZE = 75
def save_object(obj, name):

    f = open(name, 'wb')

    pickle.dump(obj, f)

    f.close()

    

def load_object(name):

    f = open(name, 'rb')

    obj = pickle.load(f)

    f.close()

    return obj
b_cat_map = load_object("../input/bonus-task-pickled-data-1/b_cat_map.pkl")

f_cat_map = load_object("../input/bonus-task-pickled-data-1/f_cat_map.pkl")

embed_fasttext = load_object("../input/bonus-task-pickled-data-1/embed_fasttext.pkl")

embed_word2vec = load_object("../input/bonus-task-pickled-data-1/embed_word2vec.pkl")

itemid = load_object("../input/bonus-task-pickled-data-1/itemid.pkl")

y_train = load_object("../input/bonus-task-pickled-data-1/y_train.pkl")

word_index = load_object("../input/bonus-x/word_index.pkl")
def build_emb_matrix(word_index, embeddings_index, dim):

    embedding_matrix = np.zeros((len(word_index) + 1, dim + 1), dtype=np.float32)

    embedding_matrix[0, dim] = 1

    for word, i in tqdm(word_index.items(), disable=not(VERBOSE)):

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i, :dim] = embedding_vector

            continue

    return embedding_matrix
%%time

embedding_matrix = np.concatenate([build_emb_matrix(word_index, embed_fasttext, FASTTEXT_DIM),

                                   build_emb_matrix(word_index, embed_word2vec, WORD2VEC_DIM)], -1)



del embed_fasttext, embed_word2vec

_ = gc.collect()

print("Embeddings memory usage", sys.getsizeof(embedding_matrix) / (1024*1024), "MB")
class SimpleReader():

    def __init__(self, file, y_train, batch_size, maxlen, index_filter=lambda x : True):

        self.fn = file

        self.file = open(file, 'r')

        self.k = 0

        self.y_train = y_train

        self.batch_size = batch_size

        self.maxlen = maxlen

        self.index_filter = index_filter

        

    def flow(self):

        X = []

        y = []

        self.k = 0

        while True:

            line = self.file.readline().replace('\n', '')

            if line is not None and len(line) > 0:

                if self.index_filter(self.k):

                    X.append(list(map(int, line.split(' '))))

                    y.append(self.y_train[self.k])

                self.k += 1

            else:

                self.file.close()

                self.file = open(self.fn, 'r')

                self.k = 0

            if len(X) == self.batch_size:

                X = pad_sequences(X, maxlen=self.maxlen, truncating='post', padding='post')

                y = np.array(y)

                yield((X, y))

                X = []

                y = []
def create_model(embedding_matrix):

    input_tensor = Input(shape=(None,))

    output_tensor = input_tensor

    output_tensor = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(output_tensor)

    trainable_embed = Embedding(embedding_matrix.shape[0], TRAINABLE_EMBED_SIZE, trainable=True)(input_tensor)

    output_tensor = Concatenate(-1)([output_tensor, trainable_embed])

    output_tensor = SpatialDropout1D(0.3)(output_tensor)

    output_tensor = Bidirectional(CuDNNLSTM(384, return_sequences=True))(output_tensor)

    output_tensor = Add()([Bidirectional(CuDNNGRU(384, return_sequences=True))(output_tensor), output_tensor])

    output_tensor = Add()([Bidirectional(CuDNNGRU(384, return_sequences=True))(output_tensor), output_tensor])

    output_tensor = Concatenate()([GlobalMaxPooling1D()(output_tensor), GlobalAveragePooling1D()(output_tensor)])

    output_tensor = Add()([Dense(384*4, activation='elu')(output_tensor), output_tensor])

    output_tensor = Dense(CATEGORY_CNT, activation='softmax')(output_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model
model = create_model(embedding_matrix)

model.summary()

K.clear_session()

gc.collect()

del model

_ = gc.collect()
%%time

train_generator = SimpleReader("../input/bonus-x-txt-fasttext-unk/X_train_fasttext_unk.txt", y_train, BATCH_SIZE, MAXLEN, index_filter=lambda x : x % 30 != 7)

val_generator = SimpleReader("../input/bonus-x-txt-fasttext-unk/X_train_fasttext_unk.txt", y_train, BATCH_SIZE, MAXLEN, index_filter=lambda x : x % 30 == 7)



lr_scheldure = LearningRateScheduler(lambda epoch, lr: EXP_DECAY_COEF ** epoch * 1e-3, verbose=VERBOSE)



model = create_model(embedding_matrix)

model.compile(optimizer=Adam(lr=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.fit_generator(

    train_generator.flow(),

    validation_data=val_generator.flow(),

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_steps=VAL_STEPS_PER_EPOCH,

    epochs=EPOCHS,

    verbose=VERBOSE,

    callbacks=[lr_scheldure]

)
predictions = np.ones((len(itemid),), dtype=np.int) * -1

gen = SimpleReader("../input/bonus-x-txt-fasttext-unk/X_test_fasttext_unk.txt", np.arange(0, len(itemid), 1), PREDICT_BATCH_SIZE, MAXLEN)

flow = gen.flow()

for i in tqdm(range((len(itemid) + PREDICT_BATCH_SIZE - 1) // PREDICT_BATCH_SIZE), disable=not(VERBOSE)):

    batch = next(flow)

    batch_predictions = model.predict_on_batch(batch[0])

    predictions[batch[1]] = batch_predictions.argmax(-1)
submission = pd.read_csv("../input/texts-classification-ml-hse-2019/sample_submission.csv")

submission['Category'] = predictions

submission['Id'] = itemid
submission['Category'] = submission['Category'].apply(lambda x : b_cat_map[x])
submission.to_csv("submission.csv", index=False)