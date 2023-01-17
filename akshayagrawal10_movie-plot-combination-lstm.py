%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
import io
df = pd.read_csv("../input/movies_genres.csv", delimiter='\t')
df
#Crime
dfuno3 = df[df.Crime == 1]
dfcrime = dfuno3[['plot']]
dfcrime
#Mystery
dfdos3 = df[df.Mystery == 1]
dfmys = dfdos3[['plot']]
dfmys
#Adventure
dftres3 = df[df.Adventure == 1]
dfadv = dftres3[['plot']]
dfadv
dfs = [dfcrime, dfmys, dfadv]
df_plot3 = pd.concat(dfs)
df_plot3.info
n_plots = len(df_plot3)
n_chars = len(' '.join(map(str, df_plot3['plot'])))

print("Plot 1 combination accounts for %d plots" % n_plots)
print("Their plots add up to %d characters" % n_chars)
sample_size = int(len(df_plot3) * 0.2)
df_plot3 = df_plot3[:sample_size]
df_plot3x = ' '.join(map(str, df_plot3['plot'])).lower()
df_plot3x[:200]
chars = sorted(list(set(df_plot3x)))
print('Count of unique characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(df_plot3x) - maxlen, step):
    sentences.append(df_plot3x[i: i + maxlen])
    next_chars.append(df_plot3x[i + maxlen])
print('Number of sequences:', len(sentences), "\n")

print(sentences[:10], "\n")
print(next_chars[:10])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(df_plot3x) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = df_plot3x[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y,
          batch_size=500,
          epochs=20,
          callbacks=[print_callback])