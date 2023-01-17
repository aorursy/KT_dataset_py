!pip install -U transformers
import os

import gc



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_mask=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen,

        truncation=True

    )

    

    return np.array(enc_di['input_ids'])
def dict_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_mask=True, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen,

        truncation=True

    )

    

    return {

        "input_ids": np.array(enc_di['input_ids']),

        "attention_mask": np.array(enc_di['attention_mask'])

    }
def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")

    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    sequence_output = transformer({"input_ids": input_word_ids, "attention_mask": attention_mask})[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs={

        "input_ids": input_word_ids,

        "attention_mask": attention_mask

    }, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

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
MAX_LEN = 192

MODEL = 'jplu/tf-xlm-roberta-large'

AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

tokenizer = AutoTokenizer.from_pretrained(MODEL)
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv').sample(frac=1.0)

n_valid_steps = valid.shape[0] // BATCH_SIZE

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
# Combine train1 with a subset of train2

train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)

]).sample(frac=1.0)

n_train_steps = train.shape[0] // BATCH_SIZE

del train1, train2

gc.collect()
!free -h
%%time 

x_train = dict_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)

y_train = train.toxic.values



del train

gc.collect()



x_valid = dict_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)

y_valid = valid.toxic.values

del valid

gc.collect()

x_test = dict_encode(test.content.values, tokenizer, maxlen=MAX_LEN)
!free -h
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(4096)

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

del x_train, x_valid, y_train, y_valid

gc.collect()
%%time

with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
# Configuration

EPOCHS = 2



train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_train_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_valid_steps,

    epochs=EPOCHS

)
model.save_weights("final_weights.h5")
sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)