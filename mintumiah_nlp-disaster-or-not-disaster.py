# Import necesssray library for the analysis and model

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

import re

import string

import pandas_profiling

import random

import string

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub





#Load 3 different files-train, test and submission

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print (train.shape, test.shape, sample_submission.shape)



# See the data attributes 

train.head()
test.head()
sample_submission.head()
train.duplicated().sum()
train = train.drop_duplicates().reset_index(drop=True)
# Let see the disaster and non-disaster frequency in traininf set

sns.countplot(y=train.target);
# Lets check with the missing data

train.isnull().sum()
test.isnull().sum()
# lets check the key word similarty in training and test data set

print (train.keyword.nunique(), test.keyword.nunique())

print (set(train.keyword.unique()) - set(test.keyword.unique()))
# Most common keywords

plt.figure(figsize=(9,6))

sns.countplot(y=train.keyword, order = train.keyword.value_counts().iloc[:15].index)

plt.title('Top 15 keywords')

plt.show()

# train.keyword.value_counts().head(10)
top_d = train.groupby('keyword').mean()['target'].sort_values(ascending=False).head(10)

top_nd = train.groupby('keyword').mean()['target'].sort_values().head(10)



plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(top_d, top_d.index, color='pink')

plt.title('Keywords with highest % of disaster tweets')

plt.subplot(122)

sns.barplot(top_nd, top_nd.index, color='yellow')

plt.title('Keywords with lowest % of disaster tweets')

plt.show()
raw_loc = train.location.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

top_only = train[train.location.isin(top_loc)]



top_l = top_only.groupby('location').mean()['target'].sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=top_l.index, y=top_l)

plt.axhline(np.mean(train.target))

plt.xticks(rotation=80)

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

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
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    batch_size=16

)



model.save('model.h5')
test_pred = model.predict(test_input)
sample_submission['target'] = test_pred.round().astype(int)

sample_submission.to_csv('sample_submission.csv', index=False)
sample_submission.head()