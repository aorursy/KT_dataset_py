# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/onion-or-not/OnionOrNot.csv')
data.shape
data.head(10)
display(data['text'][0])

display(data['text'][2])

display(data['text'][5])

display(data['text'][9])
display(data['text'][1])

display(data['text'][6])

display(data['text'][7])
import matplotlib.pyplot as plt

labels = [1,0]

plt.pie(data['label'].value_counts() )

plt.legend(labels)

plt.title('Onions or not')
import re
data_process = data.copy()
data_process.head()
data_process["text"] = data_process["text"].str.replace('[^a-zA-Z]', ' ', regex=True)
display(data['text'][2])

display(data_process['text'][2])
data_process["text"] = data_process["text"].str.lower()
display(data['text'][2])

display(data_process['text'][2])
from keras.preprocessing.text import Tokenizer
num_words = 20000

max_len = 150

emb_size = 128

X = data_process["text"]
X
token = Tokenizer(num_words = num_words)

token.fit_on_texts(list(X))
X = token.texts_to_sequences(X)
display(data['text'][2])

display(data_process['text'][2])

display(X[2])
plt.plot(X[2], label = 'sample text spectra')

plt.title(str(data['text'][2]))
from keras.preprocessing import sequence



X = sequence.pad_sequences(X, maxlen = 150)

y = pd.get_dummies(data_process['label'])
y = y.values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 42)
import nltk

from keras.models import Sequential

from keras.layers import Input,Dense, LSTM, Dropout, Flatten, Embedding, Bidirectional, GlobalMaxPool1D
def model():

    

    inp = Input(shape = (max_len, ))

    layer = Embedding(num_words, emb_size)(inp)

    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.1))(layer)

    

    layer = GlobalMaxPool1D()(layer)

    layer = Dropout(0.2)(layer)

    

    

    layer = Dense(50, activation = 'relu')(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(50, activation = 'relu')(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(50, activation = 'relu')(layer)

    layer = Dropout(0.2)(layer)

    

    

    layer = Dense(2, activation = 'softmax')(layer)

    model = Model(inputs = inp, outputs = layer)

    

    

    model.compile(loss = 'binary_crossentropy', optimizer = 'nadam', metrics=['accuracy'])

    return model
from keras.models import Model

from keras.utils import plot_model

import matplotlib.image as mpimg



model = model()

model.summary()



plot_model(model, to_file='onion.png',show_shapes=True, show_layer_names=True)

plt.figure(figsize = (30,20))

img = mpimg.imread('/kaggle/working/onion.png')

imgplot = plt.imshow(img)
from keras.callbacks import EarlyStopping, ModelCheckpoint



file_path = 'save.hd5'

checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', save_best_only=True)

early_stop = EarlyStopping(monitor = 'loss', patience = 1)
history = model.fit(x_train, y_train, batch_size = 32, epochs = 3, validation_split = 0.1, callbacks = [checkpoint,early_stop])
val_loss = history.history['val_loss']

loss = history.history['loss']
print('validation loss: ', val_loss[-1])

print('training loss: ', loss[-1])
score = model.evaluate(x_test, y_test)

print(model.metrics_names)

print(score)

print('test loss: ', score[0])

print('test accuracy: ', score[1])
history.history