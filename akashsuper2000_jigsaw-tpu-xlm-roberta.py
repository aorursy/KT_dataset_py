import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import math

import os

import re

import time



from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.metrics import AUC



from kaggle_datasets import KaggleDatasets



import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):



    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
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



EPOCHS = 20

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 200

MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
train = pd.read_csv('/kaggle/input/toxic-comment-classification/train.csv')



valid = pd.read_csv('/kaggle/input/toxic-comment-classification/validation.csv')



test = pd.read_csv('/kaggle/input/toxic-comment-classification/test.csv')
%%time 



x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test.comment_text.values, tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
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
%%time



with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
filepath = "roberta_model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')



early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)



reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, min_lr=1e-5)
n_steps = x_train.shape[0] // BATCH_SIZE



train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    callbacks=[early, checkpoint, reduce_lr]

)
y_true = test['toxic'].values

y_pred = model.predict(test_dataset).ravel()
roc_auc_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0)

plt.plot(tpr,fpr)



plt.show()