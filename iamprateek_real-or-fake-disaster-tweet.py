# previous kernel using BERT I got 83% accuracy

# in this kernel I will try to improve it
from numpy.random import seed

seed(42)

import tensorflow as tf

tf.random.set_seed(42)



from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_df.head()
train_df.info()
train_df.shape
test_df.shape
test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
# train_df['keyword'][train_df.keyword.isnull()] = 'no_keyword'

# train_df['location'][train_df.location.isnull()] = 'no_location'

# test_df['keyword'][test_df.keyword.isnull()] = 'no_keyword'

# test_df['location'][test_df.location.isnull()] = 'no_location'
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
Dropout_num = 0

learning_rate = 7e-6

valid = 0.2

epochs_num = 3

batch_size_num = 16

target_corrected = False

target_big_corrected = False
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    if Dropout_num == 0:

        out = Dense(1, activation='sigmoid')(clf_output)

    else:

        # With Dropout(Dropout_num), Dropout_num > 0

        x = Dropout(Dropout_num)(clf_output)

        out = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
if target_corrected:

    ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

    train_df.loc[train_df['id'].isin(ids_with_target_error),'target'] = 0

    train_df[train_df['id'].isin(ids_with_target_error)]
# Loading tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Encoding the text into tokens, masks, and segment flags

train_input = bert_encode(train_df.text.values, tokenizer, max_len=160)

test_input = bert_encode(test_df.text.values, tokenizer, max_len=160)

train_labels = train_df.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=5,

    batch_size=8

)



model.save('model.h5')
test_pred = model.predict(test_input)
sub['target'] = test_pred.round().astype(int)

sub.to_csv('submission.csv', index=False)

sub.head()