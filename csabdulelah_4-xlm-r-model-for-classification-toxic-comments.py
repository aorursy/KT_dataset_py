import os
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
    """
    Function to encode the word
    """
    # encode the word to vector of integer
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):
    """
    This function to build and compile Keras model
    
    """
    #Input: for define input layer
    #shape is vector with 512-dimensional vectors
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids") # name is optional 
    sequence_output = transformer(input_word_ids)[0]
    # to get the vector
    cls_token = sequence_output[:, 0, :]
    # define output layer
    out = Dense(1, activation='sigmoid')(cls_token)
    
    # initiate the model with inputs and outputs
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC()])
    
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

# input pipeline that delivers data for the next step before the current step has finished.
# The tf.data API helps to build flexible and efficient input pipelines.
# This document demonstrates how to use the tf.data 
# API to build highly performant TensorFlow input pipelines.
AUTO = tf.data.experimental.AUTOTUNE

# upload data into google cloud storage
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-large'
# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
#applied regular_encode() function to all data set
%%time 

x_train = regular_encode(train1.comment_text.values, tokenizer, maxlen=MAX_LEN)
x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values

# Create a source dataset from your input data.
# Apply dataset transformations to preprocess the data.
# Iterate over the dataset and process the elements.

train_dataset = (
    tf.data.Dataset # create dataset
    .from_tensor_slices((x_train, y_train)) # Once you have a dataset, you can apply transformations 
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)# Combines consecutive elements of this dataset into batches.
    .prefetch(AUTO) #This allows later elements to be prepared while the current element is being processed.
)

valid_dataset = (
    tf.data.Dataset # create dataset
    .from_tensor_slices((x_valid, y_valid)) # Once you have a dataset, you can apply transformations 
    .batch(BATCH_SIZE) #Combines consecutive elements of this dataset into batches.
    .cache()
    .prefetch(AUTO)#This allows later elements to be prepared while the current element is being processed.
)

test_dataset = (
    tf.data.Dataset# create dataset
    .from_tensor_slices(x_test) # Once you have a dataset, you can apply transformations 
    .batch(BATCH_SIZE)
)
%%time
with strategy.scope():
    # load model and build it
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
# fit the model on train data
n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS +3
)
# fit model on validation
n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS * 3
)
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)
# save model weights 
model.save_weights('xlm_weights.h5')