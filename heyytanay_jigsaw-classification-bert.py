!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from tqdm.notebook import tqdm

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

import tokenization

from sklearn.model_selection import train_test_split
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
def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc
train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
train.head()
sub.head()
# Encode data for input to the model

def encode_text(texts, tokenizer, maxlen=512):

    all_masks, all_tokens, all_segments = [], [], []

    

    for text in tqdm(texts):

        text = tokenizer.tokenize(text)

        text = text[:maxlen-2]

        input_sq = ["[CLS]"] + text + ["[SEP]"]

        pad_len = maxlen - len(input_sq)

        tokens = tokenizer.convert_tokens_to_ids(input_sq)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sq) + [0] * pad_len

        segment_ids = [0] * maxlen

        

        all_masks.append(pad_masks)

        all_tokens.append(tokens)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
ATO = tf.data.experimental.AUTOTUNE

nb_epochs = 1

batch_size = 16 * strategy.num_replicas_in_sync

maxLen = 192
%%time

url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(url, trainable=True)
# Get the vocab file (for tokenizing) and tokenizer itself

vocab_fl = bert_layer.resolved_object.vocab_file.asset_path.numpy()

lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_fl, lower_case)
train_input = encode_text(train['comment_text'].values, tokenizer, maxlen=192)

train_labels = train['toxic'].values

valid_input = encode_text(valid['comment_text'].values, tokenizer, maxlen=192)

valid_labels = valid['toxic'].values
test_dataset = encode_text(test['content'].values, tokenizer, maxlen=192)
def build_model(transformer, max_len=512):

    # Naming your keras ops is very important 😉

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name='input_mask')

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name='segment_ids')

    # Get the sequence output

    _, seq_op = transformer([input_word_ids, input_mask, segment_ids])

    # Get the respective class token from that sequence output

    class_tkn = seq_op[:, 0, :]

    # Final Neuron (for Classification)

    op = Dense(1, activation='sigmoid')(class_tkn)

    # Bind the inputs and outputs together into a Model

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=op)

    

    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model(bert_layer, max_len=192)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_data=(valid_input, valid_labels),

    epochs=1,

    callbacks=[checkpoint],

    batch_size=16

)
sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv("submission.csv", index=False)
model.save("final_model.h5")