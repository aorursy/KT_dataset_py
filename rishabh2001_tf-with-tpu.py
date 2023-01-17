import os

import numpy as np

import pandas as pd

import gc

import math

import itertools 

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold

import logging

# no extensive logging 

logging.getLogger().setLevel(logging.NOTSET)



import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from transformers import TFAutoModel, AutoTokenizer, TFRobertaModel

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
BATCH_SIZE = 16

MAX_LEN = 192

LR_TRANSFORMER = 5e-6

LR_HEAD = 1e-3

MODEL = 'jplu/tf-xlm-roberta-large'

AUTO = tf.data.experimental.AUTOTUNE

tokenizer = AutoTokenizer.from_pretrained(MODEL)
def connect_to_TPU():

    """Detect hardware, return appropriate distribution strategy"""

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        tpu = None



    if tpu:

        tf.config.experimental_connect_to_cluster(tpu)

        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)

    else:

        strategy = tf.distribute.get_strategy()



    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync



    return tpu, strategy, global_batch_size





tpu, strategy, global_batch_size = connect_to_TPU()

print("REPLICAS: ", strategy.num_replicas_in_sync)
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
def create_dist_dataset(X, y=None, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(X)



    if y is not None:

        dataset_y = tf.data.Dataset.from_tensor_slices(y)

        dataset = tf.data.Dataset.zip((dataset, dataset_y))



    if training:

        dataset = dataset.shuffle(len(X)).repeat()

        

    dataset = dataset.batch(global_batch_size).prefetch(AUTO)

        

    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    return dist_dataset
train_df = pd.read_csv('../input/multilingual/train.csv')

print(len(train_df))

train_df.head()
X_train = regular_encode(train_df.comment_text.values, tokenizer, MAX_LEN)

y_train = train_df.toxic.values.reshape(-1,1)
train_dist_dataset = create_dist_dataset(X_train.astype(np.int32), y_train.astype(np.float32), training= True)
val_df = pd.read_csv('../input/multilingual/val.csv')

print(len(val_df))

val_df.head()
X_val = regular_encode(val_df.comment_text.values, tokenizer, MAX_LEN)

y_val = val_df.toxic.values.reshape(-1,1)
val_dist_dataset = create_dist_dataset(X_val.astype(np.int32))
test_df = pd.read_csv('../input/multilingual/test.csv')

print(len(test_df))

test_df.head()
X_test = regular_encode(test_df.comment_text.values, tokenizer, MAX_LEN)
test_dist_dataset = create_dist_dataset(X_test.astype(np.int32))
def get_model():

    

    transformer = TFRobertaModel.from_pretrained(MODEL)

    

    inp = Input(shape=(192,),dtype = 'int32', name = 'input_word_ids')

    x = transformer(inp)[0]

    cls_token = Dropout(0.2)(x[:, 0, :])

    

    x = Dense(256, activation = 'relu', input_shape = (1024,), name = 'custom')(cls_token)

    x = Dropout(0.2)(x)

    x = Dense(1, activation='sigmoid',input_shape = (256,), name = 'custom_1')(x)

    

    model = Model(inp, x)

    return model
def create_model_and_optimizer():

    with strategy.scope():                

        model = get_model()

        optimizer_transformer = Adam(learning_rate=LR_TRANSFORMER)

        optimizer_head = Adam(learning_rate=LR_HEAD)

    return model, optimizer_transformer, optimizer_head



model, optimizer_transformer, optimizer_head = create_model_and_optimizer()

model.summary()
def define_losses_and_metrics():

    with strategy.scope():

        loss_object = tf.keras.losses.BinaryCrossentropy(

                          from_logits=False,

                          reduction=tf.keras.losses.Reduction.NONE,

                          label_smoothing = 0.1)



        def compute_loss(labels, predictions):

            per_example_loss = loss_object(labels, predictions)

            loss = tf.nn.compute_average_loss(

                per_example_loss, global_batch_size = global_batch_size)

            return loss



        train_accuracy_metric = tf.keras.metrics.AUC(name='training_AUC')



    return compute_loss, train_accuracy_metric



compute_loss, train_accuracy_metric = define_losses_and_metrics()
@tf.function

def distributed_train_step(data):

    strategy.experimental_run_v2(train_step, args=(data,))



def train_step(inputs):

    features, labels = inputs

    

    ### get transformer and head separate vars

    # get rid of pooler head with None gradients

    transformer_trainable_variables = [ v for v in model.trainable_variables 

                                       if (('pooler' not in v.name)  and 

                                           ('custom' not in v.name))]

    head_trainable_variables = [ v for v in model.trainable_variables 

                                if 'custom'  in v.name]



    # calculate the 2 gradients ( note persistent, and del)

    with tf.GradientTape(persistent=True) as tape:

        predictions = model(features, training=True)

#         print(labels, predictions)

#         labels = tf.reshape(labels, (16,1))

#         tf.expand_dims(labels, -1)

        loss = compute_loss(labels, predictions)

    gradients_transformer = tape.gradient(loss, transformer_trainable_variables)

    gradients_head = tape.gradient(loss, head_trainable_variables)

    del tape

        

    ### make the 2 gradients steps

    optimizer_transformer.apply_gradients(zip(gradients_transformer, 

                                              transformer_trainable_variables))

    optimizer_head.apply_gradients(zip(gradients_head, 

                                       head_trainable_variables))



    train_accuracy_metric.update_state(tf.round(labels), predictions)



def predict(dataset):  

    predictions = []

    for tensor in dataset:

        predictions.append(distributed_prediction_step(tensor))

    predictions = np.vstack(list(map(np.vstack,predictions)))

    return predictions



@tf.function

def distributed_prediction_step(data):

    predictions = strategy.experimental_run_v2(prediction_step, args=(data,))

    return strategy.experimental_local_results(predictions)



def prediction_step(inputs):

    features = inputs  # note datasets used in prediction do not have labels

    predictions = model(features, training=False)

    return predictions
def train(train_dist_dataset, val_dist_dataset=None, y_val=None, total_steps=2000, validate_every=200):

    train_accuracy_metric.reset_states()

    best_weights, history = None, []

    step = 0

    ### Training loop ###

    for tensor in train_dist_dataset:

        distributed_train_step(tensor) 

        step+=1



        if (step % validate_every == 0):   

            ### Print train metrics ###  

            train_metric = train_accuracy_metric.result().numpy()

            print("Step %d, train AUC: %.5f" % (step, train_metric))   

            

            ### Test loop with exact AUC ###

            if val_dist_dataset:

                val_metric = roc_auc_score(np.round(y_val), predict(val_dist_dataset))

                print("Step %d,   val AUC: %.5f" %  (step,val_metric))   

                

                # save weights if it is the best yet

                history.append(val_metric)

                if history[-1] == max(history):

                    best_weights = model.get_weights()



            ### Reset (train) metrics ###

            train_accuracy_metric.reset_states()

            

        if step  == total_steps:

            break

    

    ### Restore best weighths ###

    if(best_weights):

        model.set_weights(best_weights)
train(train_dist_dataset, val_dist_dataset, y_val, total_steps = 1000, validate_every = 100)
preds = predict(test_dist_dataset)[:,0]