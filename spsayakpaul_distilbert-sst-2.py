# Reference - https://www.kaggle.com/docs/tpu

from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
import tensorflow as tf

print(tf.__version__)
# Configure TPU

# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)





# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



BATCH_SIZE = 32 * tpu_strategy.num_replicas_in_sync
import os

import time

import numpy as np

import pandas as pd

import transformers

import matplotlib.pyplot as plt



%matplotlib inline



# fix random seed for reproducibility

seed = 42

np.random.seed(seed)

tf.random.set_seed(seed)
data_dir = tf.keras.utils.get_file(

      fname='SST-2.zip',

      origin='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',

      extract=True)

data_dir = os.path.join(os.path.dirname(data_dir), 'SST-2')
train = os.path.join(data_dir, "train.tsv")

valid = os.path.join(data_dir, "dev.tsv")

test = os.path.join(data_dir, "test.tsv")
train_dataset = pd.read_csv(train, sep='\t')

valid_dataset = pd.read_csv(valid, sep='\t')

test_dataset = pd.read_csv(test, sep='\t')
train_dataset.info()
valid_dataset.info()
train_dataset.head()
train_reviews = train_dataset['sentence'].values

train_sentiments = train_dataset['label'].values



valid_reviews = valid_dataset['sentence'].values

valid_sentiments = valid_dataset['label'].values



test_reviews = test_dataset['sentence'].values



train_reviews.shape, valid_reviews.shape, test_reviews.shape
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
import tqdm



def create_bert_input_features(tokenizer, docs, max_seq_length):

    

    all_ids, all_masks = [], []

    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):

        tokens = tokenizer.tokenize(doc)

        if len(tokens) > max_seq_length-2:

            tokens = tokens[0 : (max_seq_length-2)]

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        ids = tokenizer.convert_tokens_to_ids(tokens)

        masks = [1] * len(ids)

        # Zero-pad up to the sequence length.

        while len(ids) < max_seq_length:

            ids.append(0)

            masks.append(0)

        all_ids.append(ids)

        all_masks.append(masks)

    encoded = np.array([all_ids, all_masks])

    return encoded
MAX_SEQ_LENGTH = 500



train_features_ids, train_features_masks = create_bert_input_features(tokenizer, train_reviews, 

                                                                      max_seq_length=MAX_SEQ_LENGTH)

val_features_ids, val_features_masks = create_bert_input_features(tokenizer, valid_reviews, 

                                                                  max_seq_length=MAX_SEQ_LENGTH)

#test_features = create_bert_input_features(tokenizer, test_reviews, max_seq_length=MAX_SEQ_LENGTH)

print('Train Features:', train_features_ids.shape, train_features_masks.shape)

print('Val Features:', val_features_ids.shape, val_features_masks.shape)
# Create TensorFlow datasets for better performance

train_ds = (

    tf.data.Dataset

    .from_tensor_slices(((train_features_ids, train_features_masks), train_sentiments))

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(tf.data.experimental.AUTOTUNE)

)

    

valid_ds = (

    tf.data.Dataset

    .from_tensor_slices(((val_features_ids, val_features_masks), valid_sentiments))

    .batch(BATCH_SIZE)

    .prefetch(tf.data.experimental.AUTOTUNE)

)
def get_training_model():

    inp_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name="bert_input_ids")

    inp_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name="bert_input_masks")

    inputs = [inp_id, inp_mask]



    hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')(inputs)[0]

    pooled_output = hidden_state[:, 0]    

    dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled_output)

    drop1 = tf.keras.layers.Dropout(0.25)(dense1)

    dense2 = tf.keras.layers.Dense(256, activation='relu')(drop1)

    drop2 = tf.keras.layers.Dropout(0.25)(dense2)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)





    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, 

                                               epsilon=1e-08), 

                  loss='binary_crossentropy', metrics=['accuracy'])

    return model
# Compile the model with TPU Strategy

with tpu_strategy.scope():

    model = get_training_model()

    

# Train the model

start = time.time()

model.fit(train_ds, 

    validation_data=valid_ds,

    epochs=3)

end = time.time()

print(f"Training takes {end-start} seconds.")
model.save_weights('distillbert_ft_wts.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 

                                       tf.lite.OpsSet.SELECT_TF_OPS]

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

open("distilbert_sst_tflite.tflite", "wb").write(tflite_model)
!ls -lh distilbert_sst_tflite.tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 

                                       tf.lite.OpsSet.SELECT_TF_OPS]

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

open("distilbert_sst_tflite_fp16.tflite", "wb").write(tflite_model)
!ls -lh distilbert_sst_tflite_fp16.tflite
# # ==============Representative dataset====================

# train_features_ids = train_features_ids.astype(np.int32)

# train_features_masks = train_features_masks.astype(np.int32)

# train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features_ids, 

#     train_features_masks))



# def representative_dataset_gen():

#     for feature_id, feature_mask in train_tf_dataset.take(10):

#         yield [feature_id, feature_mask]



# # ==============Conversion====================

# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 

#                                        tf.lite.OpsSet.SELECT_TF_OPS]

# converter.representative_dataset = representative_dataset_gen

# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# tflite_model = converter.convert()

# open("distilbert_sst_tflite_int.tflite", "wb").write(tflite_model)