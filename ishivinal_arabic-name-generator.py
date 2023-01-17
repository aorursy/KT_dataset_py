import pandas as pd

import numpy as np

import keras

import time

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.optimizers import RMSprop,Adam

import numpy as np

import random

import os
dataset = pd.read_csv("/kaggle/input/Arabic_Names.csv")
names = dataset.loc[:,"Arabic_Name"]
step_length = 1   

epochs = 50       

batch_size = 64    

latent_dim = 128   

dropout_rate = 0.2 

verbosity = 0     

gen_amount = 10    
input_names = []

for name in names:

    name = name.rstrip()

    input_names.append(name)
concat_names = '\n'.join(input_names).lower()



chars = sorted(list(set(concat_names)))

num_chars = len(chars)



char2idx = dict((c, i) for i, c in enumerate(chars))

idx2char = dict((i, c) for i, c in enumerate(chars))



max_sequence_length = max([len(name) for name in input_names])



print('Total chars: {}'.format(num_chars))

print('Corpus length:', len(concat_names))

print('Number of names: ', len(input_names))

print('Longest name: ', max_sequence_length)
max_sequence_length = 50
sequences = []

next_chars = []

for i in range(0, len(concat_names) - max_sequence_length, step_length):

    sequences.append(concat_names[i: i + max_sequence_length])

    next_chars.append(concat_names[i + max_sequence_length])



num_sequences = len(sequences)



for i in range(20):

    print('X=[{}]   y=[{}]'.replace('\n', ' ').format(sequences[i], next_chars[i]).replace('\n', ' '))
X = np.zeros((num_sequences, max_sequence_length, num_chars), dtype=np.bool)

Y = np.zeros((num_sequences, num_chars), dtype=np.bool)



for i, sequence in enumerate(sequences):

    for j, char in enumerate(sequence):

        X[i, j, char2idx[char]] = 1

    Y[i, char2idx[next_chars[i]]] = 1

    

print('X shape: {}'.format(X.shape))

print('Y shape: {}'.format(Y.shape))

print(X[0])

print(Y[0])
model = Sequential()

model.add(LSTM(latent_dim, 

               input_shape=(max_sequence_length, num_chars),  

               recurrent_dropout=dropout_rate))

model.add(Dense(units=num_chars, activation='softmax'))



optimizer = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer)



model.summary()
start = time.time()

print('Start training for {} epochs'.format(epochs))

history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbosity)

end = time.time()

print('Finished training - time elapsed:', (end - start)/60, 'min')
def generate_names():

    r = np.random.choice(len(concat_names)-1)

    r2 = r-max_sequence_length



    sequence = concat_names[r2:r-1] + '\n'



    new_names = []

    #print(sequence)

    while len(new_names) < 1:



        x = np.zeros((1, max_sequence_length, num_chars))

        for i, char in enumerate(sequence):

            x[0, i, char2idx[char]] = 1



        probs = model.predict(x, verbose=0)[0]

        probs /= probs.sum()

        next_idx = np.random.choice(len(probs), p=probs)   

        next_char = idx2char[next_idx]   

        sequence = sequence[1:] + next_char



        if next_char == '\n':



            gen_name = [name for name in sequence.split('\n')][1]



            if len(gen_name) > 4 and gen_name[0] == gen_name[1]:

                gen_name = gen_name[1:]



            if len(gen_name) > 4 and len(gen_name) <= 7:



                if gen_name not in input_names + new_names:

                    new_names.append(gen_name.capitalize())

                    return gen_name.capitalize()

for _ in range(20):

    print(generate_names())
