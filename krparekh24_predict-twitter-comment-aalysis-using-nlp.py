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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import re

import string

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import Embedding,GlobalAveragePooling1D,Dense,Flatten,Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from keras.callbacks import callbacks

from tensorflow.keras.callbacks import ModelCheckpoint
train_cv = pd.read_csv("../input/nlp-getting-started/train.csv")

test_cv = pd.read_csv("../input/nlp-getting-started/test.csv")
train_cv.shape
test_cv.shape
train_cv.head()
example="Link : https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
train_cv['text']=train_cv['text'].apply(lambda x : remove_URL(x))

test_cv['text']=test_cv['text'].apply(lambda x : remove_URL(x))
example = """<div>

<h1>Real or Not</h1>

<p>Kaggle</p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
train_cv['text']=train_cv['text'].apply(lambda x : remove_html(x))

test_cv['text']=test_cv['text'].apply(lambda x : remove_html(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #Engineer"

print(remove_punct(example))
train_cv['text']=train_cv['text'].apply(lambda x : remove_punct(x))

test_cv['text']=test_cv['text'].apply(lambda x : remove_punct(x))
train_cv[['text']].head()
vocab_size = 18105

emb = 32

max_length = 30

training_size = 7613

oov_tok = "<00V>" # for out of vocabulary
training_sentences = train_cv['text'].to_list()

testing_sentences = test_cv['text'].to_list()
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(train_cv['text'].to_list())

word_index = tokenizer.word_index
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(test_cv['text'].to_list())

word_index1 = tokenizer.word_index
max = 32

sequence = tokenizer.texts_to_sequences(train_cv['text'].to_list())

train_padded = pad_sequences(sequence, maxlen = max, padding = 'post')

print(train_padded)
max = 32

sequence = tokenizer.texts_to_sequences(test_cv['text'].to_list())

test_padded = pad_sequences(sequence, maxlen = max, padding = 'post')

print(test_padded)
model = tf.keras.Sequential([

        tf.keras.layers.Embedding(input_dim = vocab_size ,output_dim = emb ,input_length = max),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1, activation = 'sigmoid')

        ])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
train = train_padded[:train_cv.shape[0]]
test = test_padded[:test_cv.shape[0]]
from sklearn.model_selection import train_test_split

x_t,x_v,y_t,y_v = train_test_split(train,train_cv['target'].values,test_size=0.20)
x_t
x_v
y_t
y_v
num_epoch = 10

history = model.fit(x_t, y_t, epochs = num_epoch, validation_data = (x_v, y_v), verbose = 2)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train')

plt.plot(history.history['val_accuracy'], label = 'Validation')

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')
sample_sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_predict = model.predict(test)
sample_sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sample_sub['target'] = y_predict.round().astype(int)
sample_sub.to_csv('sample_sub.csv',index = False, header= True)
sample_sub.head()