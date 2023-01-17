#############################

# Experiment configurations #

#############################



INDO_DATA_NAME = 'prosa'

DATA_PATH_INDO = '../input/dataset-prosa'



FOREIGN_DATA_NAME = 'yelp'

DATA_PATH_FOREIGN = '../input/yelp-review-dataset'



MODEL_NAME = 'XLM_R'



EXPERIMENT_TYPE = 'C' # A / B / C

TOTAL_DATA = 10981 # 500 / 750 / 1000 / 2500 / 5000 / 7500 / 10981

FOREIGN_LANG_DATA_MULT = 3 # 0.5 / 1 / 1.5 / 2 / 3

RANDOM_SEED = 1

VALIDATION_DATA = 0.1

EPOCHS = 25

LEARNING_RATE = 1e-5

USE_TPU = True
from model_full import set_seed, regular_encode, build_model, callback

from load_data import load_experiment_dataset

import os

import random

import pandas as pd

import numpy as np

import torch

import tensorflow as tf

import transformers

from transformers import TFAutoModel, AutoTokenizer



set_seed(seed=RANDOM_SEED)
if USE_TPU:

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

    BATCH_SIZE = 8 * strategy.num_replicas_in_sync

    

else:

    BATCH_SIZE = 8 * 8



print("REPLICAS: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE

MAX_LEN = 512



if MODEL_NAME == 'XLM_R':

    MODEL = 'jplu/tf-xlm-roberta-large'

elif MODEL_NAME == 'mBERT':

    MODEL = 'bert-base-multilingual-cased'
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_experiment_dataset(INDO_DATA_NAME,

                                                                                   FOREIGN_DATA_NAME,

                                                                                   tipe=EXPERIMENT_TYPE, 

                                                                                   total_data=TOTAL_DATA, 

                                                                                   foreign_mult=FOREIGN_LANG_DATA_MULT, 

                                                                                   valid_size=VALIDATION_DATA)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
%%time 



x_train = regular_encode(x_train, tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(x_valid, tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(x_test, tokenizer, maxlen=MAX_LEN)
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(len(x_train),

             seed=RANDOM_SEED)

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

if USE_TPU:

    with strategy.scope():

        transformer_layer = TFAutoModel.from_pretrained(MODEL)

        model = build_model(transformer_layer, max_len=MAX_LEN, learning_rate=LEARNING_RATE)

        

else:

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN, learning_rate=LEARNING_RATE)

    

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

print(n_steps)
train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    callbacks = callback(), 

    epochs=EPOCHS

)
model.save_weights('model.h5') 
result = pd.DataFrame()

result['y_pred'] = model.predict(test_dataset, verbose=1).flatten()

result['y_true'] = y_test

result.to_csv('result_{}_{}_{}_{}_{}_{}_full.csv'.format(INDO_DATA_NAME,

                                                    FOREIGN_DATA_NAME,

                                                    MODEL_NAME,

                                                    EXPERIMENT_TYPE,

                                                    TOTAL_DATA,

                                                    FOREIGN_LANG_DATA_MULT),

              index=False)