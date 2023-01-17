# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#using tensorflow and keras high-level api to build our model
import tensorflow as tf # 
import tensorflow.keras as keras
tf.logging.set_verbosity('DEBUG')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#set frac = 1.0 will randomize the whole train data
train.sample(frac=1.0)
print(train.info())
print(train.describe())
print(test.info())
print(test.describe())
#for text embedding, we will use tensorflow api word2vec from google
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import text, sequence
train_sentences = train['comment_text'].fillna('N/A').values
test_sentences = test['comment_text'].fillna('N/A').values

#Fig : sentences length distribution
all_sentences = list(train_sentences) + list(test_sentences)
all_sentences_len = [len(text.text_to_word_sequence(sentence))
                     for sentence in all_sentences]

plt.hist(all_sentences_len, bins=200)
plt.xlim(0, 500)
plt.xlabel('sentence length')
plt.ylabel('number')
plt.table(cellText=[[np.mean(all_sentences_len), np.median(all_sentences_len),
                     np.percentile(all_sentences_len, 95), np.max(all_sentences_len)]],
          rowLabels=['value'],
          colLabels=['Mean', 'Median(50%)', '95%', 'Max(100%)'],
          loc='top')
plt.show()



MAX_FEATURES = 20000
MAX_SENTENCE_LENGTH = 200
train_classes = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


#transform text into integer sentences with same length
tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(train_sentences))

train_sentences = tokenizer.texts_to_sequences(train_sentences)
test_sentences = tokenizer.texts_to_sequences(test_sentences)
train_sentences = sequence.pad_sequences(train_sentences, maxlen=MAX_SENTENCE_LENGTH)
test_sentences = sequence.pad_sequences(test_sentences, maxlen=MAX_SENTENCE_LENGTH)
def my_model(lstm_units, dropout_rate, optimizer):
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(MAX_SENTENCE_LENGTH, )),
        keras.layers.Embedding(input_dim=MAX_FEATURES, output_dim=int(MAX_FEATURES**0.25)),
        keras.layers.LSTM(units=lstm_units),
        #keras.layers.GlobalMaxPool1D(),
        keras.layers.Dense(lstm_units, activation=tf.nn.relu),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(6, activation=tf.nn.sigmoid)
    ])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )
    return model

#define some hyper parameter
batch_size = 32
epochs = 2
lstm_units = 50
dropout_rate = 0.5
optimizer = tf.train.AdamOptimizer()

#define the model
model = my_model(lstm_units, dropout_rate, optimizer)
print(model.summary())
#define some callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
checkpoint_cb = ModelCheckpoint(filepath='model_toxic/cp.ckpt',
                                save_weights_only=True,
                                mode='min',
                                 verbose=1,
                                save_best_only=True)
tensorboard_cb = TensorBoard(log_dir='model_toxic',
                             histogram_freq=1,
                             write_grads=True)

#start train and evaluate
model.fit(
    x=train_sentences,
    y=train_classes,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[checkpoint_cb, tensorboard_cb],
)