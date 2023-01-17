import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt
import sys
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
raw_data = pd.read_json('/kaggle/input/wallstreetbets-subreddit-submissions/wallstreetbets_submission.json', lines=True)
for i in range(len(raw_data.columns)):
    print(raw_data.columns[i])
# we want selftext from all DD posts
# the 'link_flair_css_class' is what we're looking for
raw_data['link_flair_css_class'].unique()
raw_data['link_flair_css_class'].isin(['dd']).sum()
# There are 18011 DD posts
# Let's filter for DD and see how many there are as a function of time
dd_posts = raw_data[raw_data['link_flair_css_class'].isin(['dd'])]
# let's remove the rest to clear up some memory
del(raw_data)
# 'created_utc' will give us a timestamp, which we can convert from unix time to a datetime
dd_posts['timestamp'] = pd.to_datetime(dd_posts['created_utc'].map(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
dd_posts['timestamp'].groupby(dd_posts.timestamp.dt.to_period("M")).agg('count').plot()
plt.ylabel('Number of DD posts')
plt.xlabel('Month')

# There was a lot of action around the March crash, unsurprisingly
text = dd_posts['selftext'].str.cat(sep=" ")[-500000:] # take last 500 thousand characters
print('text length:', len(text))
# 13 million characters

chars = sorted(list(set(text)))
print('total chars: ', len(chars))
# 416 unique characters

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback
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

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = np.random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
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
from keras.callbacks import ModelCheckpoint

filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)
callbacks = [print_callback, checkpoint, reduce_lr]
model.fit(x, y, batch_size=128, epochs=5, callbacks=callbacks)
def generate_text(length, diversity):
    # Get random starting text
    start_index = np.random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated
generate_text(100, diversity=0.4)
# It's largely gibberish right now, so we should try a more complex model and probably limit it to a smaller subset of characters
