import pandas as pd 

import numpy as np

import os



# print(os.listdir("../input"))

file1 = "../input/alice-in-wonderland/Wonderland.txt"

file2 = "../input/shakespeare-sonnet/Shakespeare.txt"

file3 = "../input/eminemlyrics/CombinedLyrics.txt"



# Perform lyric generation for using Eminem's songs

raw_text = open(file3).read()

raw_text = raw_text.lower()



# Print the first few characters for reference

print("----- First few characters for reference -----\n")

print(raw_text[0:1000])



char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 

            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']



raw_text_final = ""

for r in raw_text:

    if r in char_list:

        raw_text_final += r



n_chars = len(raw_text_final)

print("\nTotal number of characters : {}".format(n_chars))
# Create a mapping from characters to integers

unique_chars = sorted(list(set(raw_text_final)))

print("Total unique characters : " + str(len(unique_chars)))

print(unique_chars)



print("\n")



char_to_int = dict((c, i) for i, c in enumerate(unique_chars))

print(char_to_int)
# Prepare the dataset of input to output pairs encoded as integers

seq_length = 100 # Sequence length of 100 characters mapped to 1 output character

X = []

Y = []



for i in range(0, n_chars - seq_length, 1):

    seq_in = raw_text_final[i:i + seq_length]

    seq_out = raw_text_final[i + seq_length]

    X.append([char_to_int[char] for char in seq_in])

    Y.append(char_to_int[seq_out])



print("----- Length of X and Y -----")

print(len(X))

print(len(Y))



n_patterns = len(X)

print("\nTotal Patterns: {}".format(n_patterns))



for i in range(0,5):

    print(X[i])

    print(Y[i])



# for i in range(0,len(Y)):

#     if(Y[i] == 0):

#         print(i)
from keras.utils.np_utils import to_categorical



# Reshape X to be [samples, time steps, features]

dataX = np.reshape(X, (n_patterns, seq_length, 1))



# Normalize

dataX = dataX / float(len(unique_chars))



# One hot encode the output variable

dataY = np.array(to_categorical(Y))



print("----- Shapes -----")

print(dataX.shape) # Input

print(dataY.shape) # Output
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.callbacks import ModelCheckpoint



# Define and use the LSTM Model

model = Sequential()

model.add(LSTM(256, input_shape=(dataX.shape[1], dataX.shape[2])))

model.add(Dropout(0.2))



# Define the output layer

model.add(Dense(dataY.shape[1], activation='softmax'))



# Define the loss and optimizer to use

model.compile(loss='categorical_crossentropy', optimizer='adam')



# Fit the model

model.fit(dataX, dataY, epochs=10, batch_size=128, verbose=1)
# int to char mapping to get the lyric character

int_to_char = dict((i, c) for i, c in enumerate(unique_chars))

print(int_to_char)
import sys



# Pick a random seed

start = np.random.randint(0, len(X)-1)

print("Start : " + str(start))



# Display the generated pattern (Length should be 100)

pattern = X[start]

print("Length of pattern : " + str(len(pattern)))

print("Pattern : " + str(pattern))



print("\nSeed : " + ''.join([int_to_char[value] for value in pattern]))



# Generate characters

for i in range(250):

    x = np.reshape(pattern, (1, len(pattern), 1))

#     print(x.shape)

    x = x / float(len(unique_chars)) # Perform the normalisation as done for the input training data

#     print(x)

    prediction = model.predict(x, verbose=0)

#     print(prediction)

    index = np.argmax(prediction)

#     print(index)

    result = int_to_char[index]

    seq_in = [int_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)

    pattern = pattern[1:len(pattern)]

#     print("\nSeed : " + ''.join([int_to_char[value] for value in pattern]))

print("\nDone.")