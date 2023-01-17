import os, time

import pandas

import tensorflow as tf

import tensorflow_hub as hub

from kaggle_datasets import KaggleDatasets # comment this if not running on Kaggle

print(tf.version.VERSION)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection

    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

except ValueError:

    tpu = None

    gpus = tf.config.experimental.list_logical_devices("GPU")



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

elif len(gpus) > 1: # multiple GPUs in one VM

    strategy = tf.distribute.MirroredStrategy(gpus)

else: # default strategy that works on CPU and single GPU

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)



# mixed precision

# On TPU, bfloat16/float32 mixed precision is automatically used in TPU computations.

# Enabling it in Keras also stores relevant variables in bfloat16 format (memory optimization).

# On GPU, specifically V100, mixed precision must be enabled for hardware TensorCores to be used.

# XLA compilation must be enabled for this to work. (On TPU, XLA compilation is the default)

MIXED_PRECISION = True

if MIXED_PRECISION:

    if tpu: 

        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

    else: #

        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

        tf.config.optimizer.set_jit(True) # XLA compilation

    tf.keras.mixed_precision.experimental.set_policy(policy)

    print('Mixed precision enabled')
SEQUENCE_LENGTH = 128



# Note that private datasets cannot be copied - you'll have to share any pretrained models 

# you want to use with other competitors!

BERT_GCS_PATH = KaggleDatasets().get_gcs_path('bert-multi')

# BERT_GCS_PATH = gs:// ... # if using your own bucket



BERT_GCS_PATH_SAVEDMODEL = BERT_GCS_PATH + "/bert_multi_from_tfhub"



GCS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

# GCS_PATH = gs:// ... # if using your own bucket



BATCH_SIZE = 64 * strategy.num_replicas_in_sync



TRAIN_DATA = GCS_PATH + "/jigsaw-toxic-comment-train-processed-seqlen{}.csv".format(SEQUENCE_LENGTH)

TRAIN_DATA_LENGTH = 223549 # rows

VALID_DATA = GCS_PATH + "/validation-processed-seqlen{}.csv".format(SEQUENCE_LENGTH)

STEPS_PER_EPOCH = TRAIN_DATA_LENGTH // BATCH_SIZE
from transformers import TFAutoModel





def multilingual_bert_model(max_seq_length=SEQUENCE_LENGTH, trainable_bert=True):

    """Build and return a multilingual BERT model and tokenizer."""

    input_word_ids = tf.keras.layers.Input(

        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.layers.Input(

        shape=(max_seq_length,), dtype=tf.int32, name="input_mask")

    segment_ids = tf.keras.layers.Input(

        shape=(max_seq_length,), dtype=tf.int32, name="all_segment_id")

    

    # Load a SavedModel on TPU from GCS. This model is available online at 

    # https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1. You can use your own 

    # pretrained models, but will need to add them as a Kaggle dataset.

    

    #bert_layer = tf.saved_model.load(BERT_GCS_PATH_SAVEDMODEL)

    # Cast the loaded model to a TFHub KerasLayer.

    #bert_layer = hub.KerasLayer(bert_layer, trainable=trainable_bert)

    bert_layer = TFAutoModel.from_pretrained('bert-base-multilingual-uncased')

    

    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])

    output = tf.keras.layers.Dense(32, activation='relu')(pooled_output)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels', dtype=tf.float32)(output)



    return tf.keras.Model(inputs={'input_word_ids': input_word_ids,

                                  'input_mask': input_mask,

                                  'all_segment_id': segment_ids},

                          outputs=output)
with strategy.scope():

    multilingual_bert = multilingual_bert_model()



    # Compile the model. Optimize using stochastic gradient descent.

    multilingual_bert.compile(

        loss=tf.keras.losses.BinaryCrossentropy(),

        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),

        metrics=[tf.keras.metrics.AUC()])



multilingual_bert.summary()