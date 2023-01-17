import numpy as np

import pandas as pd

import string



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# Removing punctuation and numbers from the data



print(train_data.head()['text'])

chars_to_remove = list(string.punctuation) + list((str(x) for x in range(10)))



def remove_punct_cap(string):

    output = ''

    for char in string:

        if char not in chars_to_remove:

            output += char.lower()

    return output



train_data['text'] = train_data['text'].apply(remove_punct_cap)



print('\n After preprocessing: \n')

print(train_data.head()['text'])
train_data_arr = train_data.values

train_X = train_data_arr[:,:4]

train_X = np.nan_to_num(train_X)

train_y = train_data_arr[:,4:]



test_data_arr = test_data.values

test_X = np.nan_to_num(test_data_arr)
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences



def encode(data):

    data = [one_hot(data[i, 3], 1000) for i in range(len(data))]

    data = pad_sequences(data, maxlen = 33, padding = 'post')

    return data



train_X_encoded = encode(train_X)

print(train_X_encoded.shape)



test_X_encoded = encode(test_X)

print(test_X_encoded.shape)
from keras.models import Sequential

from keras import layers

from keras.layers.embeddings import Embedding



def build_model():

    model = Sequential()

    model.add(layers.Embedding(20000, 16, input_length=33))

    model.add(layers.LSTM(12, dropout=0.2, recurrent_dropout=0.2))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    return model

   

model = build_model()

model.summary()
history = model.fit(train_X_encoded, train_y, epochs=20, batch_size=128, validation_split=0.2, verbose=0)
import matplotlib.pyplot as plt

%matplotlib inline



acc = history.history['acc']

val_acc = history.history['val_acc']

epochs = [i+1 for i in range(len(acc))]
plt.plot(epochs, acc, 'o', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Accuracy')

plt.legend()

plt.show()
# Determining how many epochs to use

print(val_acc)

triple_average = [sum([val_acc[j] for j in [i, i+1, i+2]])/3 for i in range(len(acc)-3)]

print(triple_average)

epochs = [i+2 for i in range(len(acc)-3) if triple_average[i] == max(triple_average)]

print(epochs)



# Building and fitting the model



model = build_model()

history = model.fit(train_X_encoded, train_y, epochs=epochs[0], batch_size=128, verbose=0)
predictions = model.predict(test_X_encoded)

predictions = predictions.reshape(len(predictions))

predictions = [int(x) for x in np.rint(predictions)]

output = pd.DataFrame({'id': test_data.id, 'target': predictions})

output.to_csv('submission.csv', index=False)

print('Complete')