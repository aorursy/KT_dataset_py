# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/"))
print(os.listdir("../input/"))

#Training data

train = pd.read_csv('../input/nlp-getting-started/train.csv')

print('Training data shape: ', train.shape)

train.head()
!wget https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
# tensorflow imports

import tensorflow as tf

from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow_hub as hub

import tokenization
# constants

MAX_SEQ_LEN = 128  # tweets are only 240 characters so this should be enough for most cases

NUM_EPOCHS = 20

START_TOKEN = '[CLS]'

END_TOKEN = '[SEP]'
# get the bert model we want and use it to set up the appropriate tokenizer

bert_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'

bert_layer = hub.KerasLayer(bert_model_url, trainable=False)  # freeze bert layers for now to save on performance

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
def bert_tokenize(text):

    tokenized_text = tokenizer.tokenize(text)

    tokenized_text = tokenized_text[:MAX_SEQ_LEN-2]

    text_seq = [START_TOKEN] + tokenized_text + [END_TOKEN]

    padding = MAX_SEQ_LEN - len(text_seq)

    token_ids = tokenizer.convert_tokens_to_ids(text_seq)

    token_ids += [0] * padding

    pad_masks = [1] * len(text_seq) + [0] * padding

    segment_ids = [0] * MAX_SEQ_LEN

    

    return np.array(token_ids), np.array(pad_masks), np.array(segment_ids)
def define_model(bert_layer):

    input_token_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)

    input_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)

    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)

    

    _, sequence_output = bert_layer([input_token_ids, input_mask, input_segments], )

    lstm_layer = Bidirectional(LSTM(units=64, return_sequences=False))(sequence_output)

    output = Dense(units=1, activation='sigmoid')(lstm_layer)

    model = Model(inputs=[input_token_ids, input_mask, input_segments], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
# get the data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_X = np.stack(train['text'].apply(bert_tokenize), axis=1)

test_X = np.stack(test['text'].apply(bert_tokenize), axis=1)

train_y = train['target'].values
model = define_model(bert_layer)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

es = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001, verbose=1)



train_history = model.fit(

    [train_X[0, :], train_X[1, :], train_X[2, :]], train_y,

    validation_split=0.2,

    epochs=NUM_EPOCHS,

    callbacks=[checkpoint, es],

    batch_size=32)
test_y = model.predict([test_X[0, :], test_X[1, :], test_X[2, :]])
submission['target'] = test_y.round().astype(int)

submission.to_csv('submission.csv', index=False)