# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
DATA_DIR = '../input/'
TRAIN_DATA = 'train.npy'
TEST_DATA = 'test.npy'
TEST_ID_DATA = 'test_id.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
DATA_CONFIGS = 'data_configs.jsonb'
MAX_SEQUENCE_LENGTH = 31
dataset = pd.read_csv(DATA_DIR + 'train.csv')
question_text = list(dataset['question_text'])
labels = np.array(dataset['target'], dtype=np.int64)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(question_text)
sequence = tokenizer.texts_to_sequences(question_text)
train_data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_dataset = pd.read_csv(DATA_DIR + 'test.csv')
test_question_text = list(test_dataset['question_text'])
test_id = np.array(test_dataset['qid'])
test_sequence = tokenizer.texts_to_sequences(test_question_text)
test_data = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
TEST_SPLIT = 0.1
RNG_SEED = 13371447
VOCAB_SIZE = len(tokenizer.word_index) + 1
EMB_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 1

input_train, input_eval, label_train, label_eval = train_test_split(train_data, labels, test_size=TEST_SPLIT, random_state=RNG_SEED)
CONV_FEATURE_DIM = 128
CONV_WINDOW_SIZE = 3
FC_FEATURE_DIM = 128

NUM_CONV_LAYERS = 5
NUM_FC_LAYERS = 10
def mapping_fn(X, Y):
    input, label = {'x': X}, Y
    return input, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.shuffle(buffer_size=len(input_train))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_eval, label_eval))
    dataset = dataset.batch(64)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()
def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    def conv_block(inputs):
        conv_layer = tf.keras.layers.Conv1D(CONV_FEATURE_DIM, 
                                            CONV_WINDOW_SIZE,  
                                            padding='same')(inputs)

        glu_layer = tf.keras.layers.Dense(CONV_FEATURE_DIM * 2, 
                                             activation=tf.nn.relu)(conv_layer)

        scored_output, output_layer = tf.split(glu_layer, 2, axis=-1)

        output_layer = output_layer * tf.nn.sigmoid(scored_output)

        return output_layer

    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE,EMB_SIZE)(features['x'])
    embedding_layer = tf.keras.layers.Dropout(0.2)(embedding_layer)

    with tf.variable_scope('conv_layers'):
        for i in range(NUM_CONV_LAYERS):
            input_layer = conv_output_layer if i > 0 else embedding_layer
            conv_output_layer = conv_block(input_layer)
            conv_output_layer = tf.keras.layers.Dropout(0.2)(input_layer + conv_output_layer)
    
    flatten_layer = tf.keras.layers.Flatten()(conv_output_layer)
    flatten_layer = tf.keras.layers.Dense(FC_FEATURE_DIM, activation=tf.nn.relu)(flatten_layer)
    with tf.variable_scope('dense_layers'):
        for i in range(NUM_FC_LAYERS):
            input_layer = fc_output_layer if i > 0 else flatten_layer
            fc_output_layer = tf.keras.layers.Dense(FC_FEATURE_DIM, activation=tf.nn.relu)(input_layer)
            fc_output_layer = tf.keras.layers.Dropout(0.2)(input_layer + fc_output_layer)

    logits = tf.keras.layers.Dense(1)(fc_output_layer)
    
    if PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.round(tf.nn.sigmoid(logits))
            })
    
    labels = tf.reshape(labels, [-1, 1])
    
    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss = loss)
    
    if EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        f1_score = tf.contrib.metrics.f1_score(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'f1 score': f1_score, 'acc': accuracy})
est = tf.estimator.Estimator(model_fn, model_dir="model/checkpoint/cnn_model")
est.train(train_input_fn)
est.evaluate(eval_input_fn)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":test_data}, shuffle=False)
predictions = np.array([int(p['prob'][0]) for p in est.predict(input_fn=predict_input_fn)], dtype=np.int32)
output = pd.DataFrame( data={"qid": test_id, "prediction": predictions} )
output.to_csv("submission.csv", index=False, quoting=3 )
