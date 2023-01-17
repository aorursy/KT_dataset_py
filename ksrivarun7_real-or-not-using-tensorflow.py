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
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
import re

def text(text):

    text = re.sub(r'[^a-zA-Z\']', ' ', text)

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    text = text.lower()

       

    return text
train["clean_text"]=train.text.apply(lambda x: text(x))

test["clean_text"]=test.text.apply(lambda x: text(x))
vocab_size = 10000

embedding_dim = 16

max_length = 120

trunc_type='post'

oov_tok = "<OOV>"

training_size = 20000

padding_type = 'post'
train_sentences = train.clean_text

test_sentences = test.clean_text

train_labels = train.target
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

token = Tokenizer(num_words=vocab_size,oov_token = oov_tok)

token.fit_on_texts(train_sentences)

word_index = token.word_index



train_seq = token.texts_to_sequences(train_sentences)

train_pad = pad_sequences(train_seq, maxlen = max_length,truncating = trunc_type,padding = padding_type)

test_seq = token.texts_to_sequences(test_sentences)

test_pad = pad_sequences(test_seq, maxlen = max_length,truncating = trunc_type,padding = padding_type)
import tensorflow as tf

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()



import numpy as np

train_labels = np.array(train_labels)

#test_labels = np.array(test_labels)

num_epochs = 25

history = model.fit(train_pad, train_labels, epochs=num_epochs,verbose = 2)
predictions = model.predict(test_pad)



pred = np.round(predictions).astype('int').flatten()

sub = pd.DataFrame({'id':test.id,'target':pred})

#sub.to_csv('Sub1.csv',index = False)