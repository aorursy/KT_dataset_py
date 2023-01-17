import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import math

import os

import re

import time



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve



from kaggle_datasets import KaggleDatasets
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



EPOCHS = 10

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 200



MAX_FEATURES = 100000

EMBED_SIZE = 300
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'



train = pd.read_csv('/kaggle/input/toxic-comment-classification/train.csv')



valid = pd.read_csv('/kaggle/input/toxic-comment-classification/validation.csv')



test = pd.read_csv('/kaggle/input/toxic-comment-classification/test.csv')
x_train = train['comment_text'].str.lower()



y_train = train['toxic'].values



x_valid = valid['comment_text'].str.lower()



y_valid = valid['toxic'].values



x_test = test['comment_text'].str.lower()
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.x_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.x_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
%%time



tok=text.Tokenizer(num_words=MAX_FEATURES,lower=True)

tok.fit_on_texts(list(x_train)+list(x_test))



x_train=tok.texts_to_sequences(x_train)

x_test=tok.texts_to_sequences(x_test)



x_train=sequence.pad_sequences(x_train,maxlen=MAX_LEN)

x_test=sequence.pad_sequences(x_test,maxlen=MAX_LEN)
%%time



embeddings_index = {}

with open(EMBEDDING_FILE,encoding='utf8') as f:

    for line in f:

        values = line.rstrip().rsplit(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs
word_index = tok.word_index



num_words = min(MAX_FEATURES, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, EMBED_SIZE))

for word, i in word_index.items():

    if i >= MAX_FEATURES:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
with strategy.scope():



    sequence_input = Input(shape=(MAX_LEN, ))

    

    x = Embedding(MAX_FEATURES, EMBED_SIZE, weights=[embedding_matrix],trainable = False)(sequence_input)



    x = SpatialDropout1D(0.1)(x)



    x = Bidirectional(GRU(200, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)



    x = Conv1D(200, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)



    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)



    x = concatenate([avg_pool, max_pool])



    x = Dense(100, activation='relu')(x)

    x = Dropout(0.1)(x)



    preds = Dense(1, activation="sigmoid")(x)

    model = Model(sequence_input, preds)

    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-5),metrics=['accuracy'])



model.summary()
filepath = "bilstm_cnn.h5"



checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')



early = EarlyStopping(monitor="val_auc", mode="max", patience=5)
%%time



n_steps = x_train.shape[0] // BATCH_SIZE



model.fit(train_dataset, steps_per_epoch=n_steps, epochs=EPOCHS, validation_data=valid_dataset, callbacks=[early, checkpoint], verbose=1)
%%time



y_true = test['toxic'].values



y_pred = []

for i in x_test:

    y_pred.append(model.predict(np.array([i])).tolist()[0][0])

    

y_pred = np.array(y_pred)
roc_auc_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0)

plt.plot(tpr,fpr)



plt.show()