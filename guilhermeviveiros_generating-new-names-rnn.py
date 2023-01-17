import numpy as np

import requests

import pandas as pd





NationalNames = pd.read_csv("../input/us-baby-names/NationalNames.csv")

full_names = NationalNames
dataset = full_names['Name']

#dataset = pd.unique(dataset)
dataset = dataset.unique()
names = []

for i in range(len(dataset)):

    names.append(dataset[i].lower() + '\n')  
print(names)
char_to_index = dict( (chr(i+96), i-1) for i in range(1,27))

char_to_index['\n'] = 26



index_to_char = dict( (i-1, chr(i+96)) for i in range(1,27))

index_to_char[26] = '\n'



T_x = len(max(names, key=len))

m = len(names)

vocab_size = len(char_to_index)
print(char_to_index)
X = np.zeros((m, T_x, vocab_size))

Y = np.zeros((m, T_x, vocab_size))



for i in range(m):

    name = list(names[i])

    #names está estendido por cada caracter

    for j in range(len(name)):

        X[i, j, char_to_index[name[j]]] = 1

        if j < len(name)-1:

            Y[i, j, char_to_index[name[j+1]]] = 1
def make_name(model):

    name = []

    x = np.zeros((1, T_x, vocab_size))

    end = False

    i = 0

    

    while end==False:

        probs = list(model.predict(x)[0,i])

        probs = probs / np.sum(probs)

        index = np.random.choice(range(vocab_size), p=probs)

        if i == T_x-2:

            character = '\n'

            end = True

        else:

            character = index_to_char[index]

        name.append(character)

        x[0, i+1, index] = 1

        i += 1

        if character == '\n':

            end = True

    

    print(''.join(name))
import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import LSTM, Dense

from keras.callbacks import LambdaCallback


model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(128, input_shape=(T_x, vocab_size), return_sequences=True))

model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam')
def generate_name_loop(epoch, _):

    if epoch % 10 == 0:

        

        print('Names generated after epoch %d:' % epoch)



        for i in range(3):

            make_name(model)

        

        print()

        

X = X[0:5000]

Y = Y[0:5000]







#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

name_generator = LambdaCallback(on_epoch_end = generate_name_loop)

model.fit(X, Y, batch_size=24, epochs=300, callbacks = [name_generator],verbose=0)

    

    