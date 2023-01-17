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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.models import Model

from keras.layers import SimpleRNN, LSTM, GRU, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.utils import plot_model

import re

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
df = pd.read_csv('/kaggle/input/pan-tadeusz-and-beniowski-classification/tadeusz_beniowski.csv', delimiter='\t', header=None, names = ['book', 'content'], encoding='utf-16')

df.head()
df.info()
sns.countplot(df.book)

plt.xlabel('1 = Beniowski, 2 = Pan Tadeusz')

plt.title('Number of Beniowski and Pan Tadeusz lines')
X = df.content

Y = df.book

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.15)
pad = 'pre'
# taking all words into account; nr of words calculated in separate script



tok = Tokenizer(filters='!"#$%&()*+,”-./:;…»—<=>?@[\\]^_`{|}~\t\n')

tok.fit_on_texts(X_train)

max_words = len(tok.word_index) + 1

sequences = tok.texts_to_sequences(X_train)

max_len = max([len(l) for l in sequences])

sequences_matrix = sequence.pad_sequences(sequences, padding = pad, maxlen = max_len)

#max_code = max((x for i, row in enumerate(sequences) for j, x in enumerate(row)))

print(tok.word_index)
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#hist = model2.fit(codes_matrix,Y_train,batch_size=128,epochs=50,

          #validation_split=0.2)

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences, padding = pad, maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#df['content'] = df['content'].map(lambda t: t.lower())

#PERMITTED_CHARS = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSTUVWXYZŹŻ "

#df['content'] = df['content'].map(lambda t: "".join(c for c in t if c in PERMITTED_CHARS))

#df['content'] = df['content'].map(lambda t: "".join('a' if c in "aąeęioóuyAĄEĘIOÓUY" else 'b' for c in t))
seperator = ' '

text = seperator.join(df.content.to_list())



vocab = sorted(set(text))

print ('{} unique characters'.format(len(vocab)))

print(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}

print (char2idx)

idx2char = np.array(vocab)



df['codes'] = df['content'].map(lambda t: [char2idx[c] for c in t])

print(df['codes'][1])
X = df.codes

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.15)
inp_dim = len(vocab)

inp_length = max([len(l) for l in X_train])

codes_matrix = sequence.pad_sequences(X_train, padding = pad, maxlen = inp_length)
def RNN2():

    inputs = Input(name='inputs',shape=[inp_length])

    layer = Embedding(inp_dim,50,input_length=inp_length)(inputs)

    layer = SimpleRNN(64, return_sequences=True)(layer)

    layer = SimpleRNN(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model2 = RNN2()

model2.summary()

model2.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
hist = model2.fit(codes_matrix,Y_train,batch_size=128,epochs=50,

          validation_split=0.2)

#hist = model2.fit(codes_matrix,Y_train,batch_size=128,epochs=10,

          #validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_codes_matrix = sequence.pad_sequences(X_test, padding = pad, maxlen = inp_length)

accr = model2.evaluate(test_codes_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
print(hist.history)
SVG(model_to_dot(model).create(prog='dot', format='svg'))
# Plot training & validation accuracy values

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()