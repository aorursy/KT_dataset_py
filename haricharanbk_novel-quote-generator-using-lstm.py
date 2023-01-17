# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm



!pip install mitdeeplearning

import mitdeeplearning as mdl



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_json('/kaggle/input/quotes-dataset/quotes.json')
print(data.shape)

print(data.columns)
data.head()
quotes = np.array(data.Quote)

print(quotes[:10])
# storing all words to a words array

all_quotes = "\n\n".join(quotes)

all_words = sorted(set(all_quotes.split()+['\n']))

len_all_words = len(all_words)

print(len_all_words)

print('\n' in all_words)

del(all_quotes)
# storing start words for later use

start_words = sorted(set(map(lambda x: x.split()[0], quotes)))

len_start_words = len(start_words)

print(start_words[:100])
# mapping and reverse-mapping

word2idx = {u:i for i, u in enumerate(all_words)}

idx2word = np.array(all_words)
#encoding data to be able to use it for training

encoded_data = []

esc_id = word2idx['\n']



for i in quotes:

    for j in i.split():

        encoded_data.append(word2idx[j])

    encoded_data.append(esc_id)

encoded_data = np.array(encoded_data)
encoded_data.shape
# returns n_batches numbered quotes from quotes array

def get_batches(data, seq_len, n_batchs):

    idxs = np.random.choice(len(data)-seq_len, n_batchs)

    x = [np.array(data[idx:idx+seq_len]) for idx in idxs]

    y = [np.array(data[idx+1:idx+seq_len+1]) for idx in idxs]

    return (np.array(x), np.array(y))
#test

x, y = get_batches(encoded_data, 100, 4)

print(x, y, sep = '\n\n')
def create_model(len_all_words, embedding_size, batch_size, rnn_units):

    return tf.keras.models.Sequential([

        tf.keras.layers.Embedding(len_all_words, embedding_size, batch_input_shape = [batch_size, None]),

        tf.keras.layers.LSTM(rnn_units, stateful = True, return_sequences = True),

        tf.keras.layers.Dense(len_all_words)

    ])

model = create_model(len_all_words, 1024, 4, 1024)

model.summary()
pred = model(x)

print(x[0])

print(tf.squeeze(tf.random.categorical(pred[0],1)).numpy())
def calc_loss(a, e):

  return tf.keras.losses.sparse_categorical_crossentropy(a, e, from_logits=True)
LEARNING_RATE = 1E-2

LEN_ALL_WORDS = len_all_words

EMBEDDING_DIM = 1024

RNN_UNITS = 1024

BATCH_SIZE = 64

SEQ_LEN = 100



EPOCHS =1500

WEIGHTS_PATH = './mod_weights.wt'

checkpoint_prefix = os.path.join(WEIGHTS_PATH, "my_ckpt")
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

model = create_model(LEN_ALL_WORDS, EMBEDDING_DIM, BATCH_SIZE, RNN_UNITS)
@tf.function

def train_step(x, y):

    with tf.GradientTape() as tape:

        y_hat = model(x)

        loss = calc_loss(y, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss        
history = []

plotter = mdl.util.PeriodicPlotter(sec=7, xlabel='Iterations', ylabel='Loss')

for i in tqdm(range(EPOCHS)):

    x, y = get_batches(encoded_data, SEQ_LEN, BATCH_SIZE)

    history.append(train_step(x, y).numpy().mean())

    plotter.plot(history)

    if i % 100 == 0:     

        model.save_weights(checkpoint_prefix)
# temp = []

# for i in range(1000):

#     temp.append(history[i].numpy().mean())

# plt.plot(temp)

plt.plot(history)
model.save_weights(WEIGHTS_PATH)
model = create_model(LEN_ALL_WORDS, EMBEDDING_DIM, 1, RNN_UNITS)

model.load_weights(WEIGHTS_PATH)

model.build(tf.TensorShape([1, None]))
def pred_freq(model, start, length):

  generated = []



  input_ = [word2idx[start]]

  input_ = tf.expand_dims(input_, 0)



  for i in range(length):

    pred = model(input_)

    pred = tf.squeeze(pred, 0)

    pred_id = tf.random.categorical(pred, 1)

    generated.append(idx2word[pred_id[-1][0].numpy()])

    input_ = pred_id.numpy()



  return start+" ".join(generated)

novel_quotes = pred_freq(model, start_words[np.random.choice(len(start_words), 1)[0]], 1000)
print(novel_quotes)