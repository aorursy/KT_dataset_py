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
from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM
max_features = 2000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train.shape)

print(x_test.shape)
import numpy as np
from keras.preprocessing import sequence
maxlen = 80



x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)
train_lens = [len(x_train[i]) for i in range(0, x_train.shape[0])]

max_review_length = max(train_lens)

max_rl_arg = np.array(train_lens).argmax()

top_words = max(map(lambda x: max(x), x_train))+1

print(max_rl_arg)

print(max_review_length)

print(top_words)
epochs = 10

batch_size = 32
embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words, embedding_vecor_length, input_length = max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split = 0.15)
score, acc = model.evaluate(x_test, y_test)

print('Test score:', score)

print('Test accuracy:', acc)
import matplotlib.pyplot as plt
def plot_fit_res(hist):

    plt.figure(figsize=plt.figaspect(0.5))

    plt.subplot(1, 2, 1)

    l = range(0, len(hist.history['val_loss']))

    plt.plot(l, hist.history['val_loss'])

    plt.title('val_loss')

    plt.subplot(1, 2, 2)

    l = range(0, len(hist.history['val_accuracy']))

    plt.plot(l, hist.history['val_accuracy'])

    plt.title('val_accuracy')
plot_fit_res(hist)
from keras.layers import Conv1D, MaxPooling1D
model_conv_mp = Sequential()

model_conv_mp.add(Embedding(top_words, embedding_vecor_length, input_length = max_review_length))

model_conv_mp.add(Conv1D(filters=max_review_length, kernel_size=3, activation='relu', padding = 'same'))

model_conv_mp.add(MaxPooling1D(pool_size = 2))

model_conv_mp.add(LSTM(100))

model_conv_mp.add(Dense(1, activation = 'sigmoid'))

model_conv_mp.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])



hist_conv_mp = model_conv_mp.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split = 0.15)

score_conv_mp, acc_conv_mp = model_conv_mp.evaluate(x_test, y_test)

print('Test score:', score_conv_mp)

print('Test accuracy:', acc_conv_mp)

plot_fit_res(hist_conv_mp)
print('Diff losses', abs(score_conv_mp-score))

print('Diff accurs', abs(acc_conv_mp-acc))
dp_k = 0.25
model_dp = Sequential()

model_dp.add(Embedding(top_words, embedding_vecor_length, input_length = max_review_length))

model_dp.add(LSTM(100, dropout=dp_k, recurrent_dropout=dp_k))

model_dp.add(Dense(1, activation = 'sigmoid'))

model_dp.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])



hist_dp = model_dp.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split = 0.15)

score_dp, acc_dp = model_dp.evaluate(x_test, y_test)

print('Test score:', score_dp)

print('Test accuracy:', acc_dp)

plot_fit_res(hist_dp)
print('Diff losses', abs(score_dp-score))

print('Diff accurs', abs(acc_dp-acc))
from keras.layers import Dropout
model_conv_dp = Sequential()

model_conv_dp.add(Embedding(top_words, embedding_vecor_length, input_length = max_review_length))

model_conv_dp.add(Conv1D(filters=max_review_length, kernel_size=3, activation='relu'))

model_conv_dp.add(MaxPooling1D(pool_size = 2))

model_conv_dp.add(Dropout(dp_k))

model_conv_dp.add(LSTM(100, dropout=dp_k, recurrent_dropout = dp_k))

model_conv_dp.add(Dense(1, activation = 'sigmoid'))

model_conv_dp.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])



hist_conv_dp = model_conv_dp.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split = 0.15)

score_conv_dp, acc_conv_dp = model_conv_dp.evaluate(x_test, y_test)

print('Test score:', score_conv_dp)

print('Test accuracy:', acc_conv_dp)

plot_fit_res(hist_conv_dp)
print('Diff losses', abs(score_conv_dp-score_conv_mp))

print('Diff accurs', abs(acc_conv_dp-acc_conv_mp))
from keras import optimizers
model_dp_lr = Sequential()

model_dp_lr.add(Embedding(top_words, embedding_vecor_length, input_length = max_review_length))

model_dp_lr.add(LSTM(100, dropout=dp_k, recurrent_dropout=dp_k))

model_dp_lr.add(Dense(1, activation = 'sigmoid'))

optim_lr_0 = optimizers.Adam(lr=0.01)

model_dp_lr.compile(loss='binary_crossentropy', optimizer = optim_lr_0, metrics=['accuracy'])



hist_dp_lr = model_dp_lr.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split = 0.15)

score_dp_lr, acc_dp_lr = model_dp_lr.evaluate(x_test, y_test)

print('Test score:', score_dp_lr)

print('Test accuracy:', acc_dp_lr)

plot_fit_res(hist_dp_lr)
model_conv_dp_lr = Sequential()

model_conv_dp_lr.add(Embedding(top_words, embedding_vecor_length, input_length = max_review_length))

model_conv_dp_lr.add(Conv1D(filters=max_review_length, kernel_size=3, activation='relu'))

model_conv_dp_lr.add(MaxPooling1D(pool_size = 2))

model_conv_dp_lr.add(Dropout(dp_k))

model_conv_dp_lr.add(LSTM(100, dropout=dp_k, recurrent_dropout = dp_k))

model_conv_dp_lr.add(Dense(1, activation = 'sigmoid'))

optim_lr = optimizers.Adam(lr=0.01)

model_conv_dp_lr.compile(loss='binary_crossentropy', optimizer = optim_lr, metrics=['accuracy'])



hist_conv_dp_lr = model_conv_dp_lr.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split = 0.15)

score_conv_dp_lr, acc_conv_dp_lr = model_conv_dp_lr.evaluate(x_test, y_test)

print('Test score:', score_conv_dp_lr)

print('Test accuracy:', acc_conv_dp_lr)

plot_fit_res(hist_conv_dp_lr)